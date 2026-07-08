import os
import sys
import json
import argparse
import chromadb
from PIL import Image
import ollama
from pydantic import BaseModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from rich import print as rprint

# Define the structured schema for the LLM
class SearchParameters(BaseModel):
    search_term: str
    project_filter: str | None

# Initialize ChromaDB and dependencies globally
client = chromadb.PersistentClient(path="./chroma_db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.get_or_create_collection(
    name="project_images", 
    embedding_function=embedding_function,
    data_loader=image_loader
)

def parse_query_with_llm(user_prompt: str, valid_projects: list) -> SearchParameters:
    """Extracts search terms and maps the requested project to a strict valid list."""
    system_instruction = (
        "You are an expert AI assistant for a local image search pipeline.\n"
        "Your task is to isolate what object/file description the user wants to see, "
        "and which project subfolder it belongs to.\n\n"
        f"CRITICAL RULES:\n"
        f"1. The ONLY allowed project folder names are: {valid_projects}.\n"
        f"2. You MUST select one of these exact strings for 'project_filter'. Do not capitalize or change them.\n"
        f"3. Do not let the project name bleed into the 'search_term'. Keep them separated.\n"
        f"4. If no valid project match is requested, set 'project_filter' to null.\n\n"
        "Example: 'Show me an image of sample from biotronik'\n"
        "Output: {\"search_term\": \"sample\", \"project_filter\": \"biotronik\"}"
    )

    response = ollama.chat(
        model='llama3', 
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        format=SearchParameters.model_json_schema()
    )
    
    result_json = json.loads(response['message']['content'])
    return SearchParameters(**result_json)

def handle_ingest(base_folder: str):
    """Scans local project directories and builds/updates the vector database."""
    supported = ('.png', '.jpg', '.jpeg', '.webp')
    if not os.path.exists(base_folder):
        rprint(f"[red][ERROR] Directory '{base_folder}' does not exist.[/red]")
        sys.exit(1)

    rprint(f"[cyan]Scanning '{base_folder}' for images...[/cyan]")
    ids, uris, metadatas = [], [], []
    
    for project_name in os.listdir(base_folder):
        project_path = os.path.join(base_folder, project_name)
        if os.path.isdir(project_path):
            for file_name in os.listdir(project_path):
                if file_name.lower().endswith(supported):
                    file_path = os.path.abspath(os.path.join(project_path, file_name))
                    
                    ids.append(file_path)
                    uris.append(file_path)
                    metadatas.append({"project": project_name, "path": file_path})

    if ids:
        collection.add(ids=ids, uris=uris, metadatas=metadatas)
        rprint(f"[green][SUCCESS] Indexed {len(ids)} images successfully into ChromaDB.[/green]")
    else:
        rprint("[red][WARNING] No supported images found to index.[/red]")

def handle_query(human_query: str):
    """Uses a dynamic baseline path to find valid projects, runs LLM parsing with 
    strict text cleanup, and retrieves images using a hybrid case-insensitive 
    filename-first + vector-fallback search approach."""
    
    # 1. Gather all data currently inside the database
    all_data = collection.get(include=['metadatas', 'uris'])
    all_metadata = all_data.get('metadatas', [])
    all_uris = all_data.get('uris', [])
    
    # Get original project folder names from database
    db_projects = list(set([m['project'] for m in all_metadata if m and 'project' in m]))

    if not db_projects:
        rprint("[red][WARNING] The vector database is empty. Please run 'ingest' first.[/red]")
        return

    # ---- CASE SENSITIVITY ----
    # Create a mapping dictionary: {'skon': 'SKON', 'biotronik': 'biotronik'}
    # This allows us to map any LLM output back to the EXACT folder name in the database.
    project_case_mapper = {p.lower(): p for p in db_projects}
    
    rprint("[cyan]Parsing intent with local LLM...[/cyan]")
    # Feed lowercase names to the LLM to standardize expectations
    parsed_args = parse_query_with_llm(human_query, list(project_case_mapper.keys()))
    
    # ---- STEP 2: PYTHON CLEANUP SAFEGUARD ----
    raw_search = parsed_args.search_term.lower().strip()
    llm_project = parsed_args.project_filter.lower().strip() if parsed_args.project_filter else None
    
    # Map the LLM's lowercase response back to the true database casing
    project_filter = project_case_mapper.get(llm_project) if llm_project else None
    
    # Remove leading filler phrases safely
    filler_phrases = ["picture of", "image of", "photo of", "screenshot of", "a view of", "show me"]
    for phrase in filler_phrases:
        if raw_search.startswith(phrase):
            raw_search = raw_search.replace(phrase, "").strip()
            
    # Safely strip out standalone words anywhere
    words = raw_search.split()
    cleaned_words = [w for w in words if w not in ["image", "photo", "picture", "a", "an", "the"]]
    search_term = " ".join(cleaned_words).strip()
    
    if not search_term:
        search_term = raw_search

    rprint(f"[cyan]-> Extracted Search Term (Cleaned): '{search_term}'[/cyan]")
    rprint(f"[cyan]-> Extracted Project Filter (Case Corrected): '{project_filter}'[/cyan]")
    
    # ---- STEP 3: HARD STRING FILENAME MATCH ----
    filename_matches = []
    for uri, meta in zip(all_uris, all_metadata):
        if not uri or not meta:
            continue
            
        # Enforce case-corrected project filter
        if project_filter and meta.get('project') != project_filter:
            continue
            
        filename = os.path.basename(uri).lower()
        
        # Look for literal string intersection
        if search_term in filename:
            filename_matches.append((uri, meta))

    if filename_matches:
        best_match_id, metadata = filename_matches[0]
        rprint(f"\n[green][SUCCESS] Exact Filename Match Found![/green]")
        print(f"Citation Path: {metadata['path']}\n")
        Image.open(best_match_id).show()
        return

    # ---- STEP 4: SEMANTIC VECTOR SEARCH (Fallback) ----
    rprint("[yellow]No exact filename match. Falling back to semantic visual search...[/yellow]")
    where_clause = {"project": project_filter} if project_filter else None
    
    results = collection.query(
        query_texts=[search_term],
        n_results=1,
        where=where_clause
    )
    
    if results['ids'] and results['ids'][0]:
        best_match_id = results['ids'][0][0]
        metadata = results['metadatas'][0][0]
        
        rprint(f"\n[green][SUCCESS] Semantic Match Found![/green]")
        print(f"Citation Path: {metadata['path']}\n")
        Image.open(best_match_id).show()
    else:
        rprint("\n[red][WARNING] No matching images found in the database.[/red]\n")

def main():
    parser = argparse.ArgumentParser(description="CLI Image Retrieval Pipeline with ChromaDB and Local LLM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command setup
    ingest_parser = subparsers.add_parser("ingest", help="Scan folders and ingest images into the vector database")
    ingest_parser.add_argument("path", type=str, help="Path to the 'projects' root folder")

    # Query command setup
    query_parser = subparsers.add_parser("search", help="Run a semantic natural language image search query")
    query_parser.add_argument("text", type=str, help="The natural language query string wrapped in quotes")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args.path)
    elif args.command == "search":
        handle_query(args.text)

if __name__ == "__main__":
    main()

