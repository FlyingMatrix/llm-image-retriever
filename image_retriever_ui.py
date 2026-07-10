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


class SearchParameters(BaseModel):
    search_term: str
    project_filter: str | None

client = chromadb.PersistentClient(path="./chroma_db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.get_or_create_collection(
    name="project_images", 
    embedding_function=embedding_function,
    data_loader=image_loader
)

def parse_query_with_llm(user_prompt: str, valid_projects: list) -> SearchParameters:
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
    # Leaving ingest untouched since it's only called directly via CLI
    supported = ('.png', '.jpg', '.jpeg', '.webp')
    if not os.path.exists(base_folder):
        print(f"[ERROR] Directory '{base_folder}' does not exist.")
        sys.exit(1)

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
        print(f"[SUCCESS] Indexed {len(ids)} images.")
    else:
        print("[WARNING] No supported images found.")

def handle_query(human_query: str):
    all_data = collection.get(include=['metadatas', 'uris'])
    all_metadata = all_data.get('metadatas', [])
    all_uris = all_data.get('uris', [])
    
    db_projects = list(set([m['project'] for m in all_metadata if m and 'project' in m]))

    if not db_projects:
        print(json.dumps({"status": "error", "message": "The vector database is empty. Please run 'ingest' first."}))
        return

    project_case_mapper = {p.lower(): p for p in db_projects}
    
    parsed_args = parse_query_with_llm(human_query, list(project_case_mapper.keys()))
    
    raw_search = parsed_args.search_term.lower().strip()
    llm_project = parsed_args.project_filter.lower().strip() if parsed_args.project_filter else None
    
    project_filter = project_case_mapper.get(llm_project) if llm_project else None
    
    filler_phrases = ["picture of", "image of", "photo of", "screenshot of", "a view of", "show me"]
    for phrase in filler_phrases:
        if raw_search.startswith(phrase):
            raw_search = raw_search.replace(phrase, "").strip()
            
    words = raw_search.split()
    cleaned_words = [w for w in words if w not in ["image", "photo", "picture", "a", "an", "the"]]
    search_term = " ".join(cleaned_words).strip()
    
    if not search_term:
        search_term = raw_search
    
    # 1. Exact Filename Match Phase
    filename_matches = []
    for uri, meta in zip(all_uris, all_metadata):
        if not uri or not meta:
            continue
            
        if project_filter and meta.get('project') != project_filter:
            continue
            
        filename = os.path.basename(uri).lower()
        if search_term in filename:
            filename_matches.append((uri, meta))

    if filename_matches:
        best_match_id, metadata = filename_matches[0]
        print(json.dumps({"status": "success", "type": "exact", "citation": metadata['path'], "image_path": best_match_id}))
        return

    # 2. Semantic Search Phase (with thresholding)
    where_clause = {"project": project_filter} if project_filter else None
    
    # Make sure we ask ChromaDB to return 'distances'
    results = collection.query(
        query_texts=[search_term],
        n_results=1,
        where=where_clause,
        include=['metadatas', 'distances'] 
    )
    
    # Check if a result exists
    if results['ids'] and results['ids'][0]:
        best_match_id = results['ids'][0][0]
        metadata = results['metadatas'][0][0]
        distance = results['distances'][0][0]
        
        # ENFORCE DISTANCE THRESHOLD: 0.0 to 0.5 usually means highly relevant or identical.
        DISTANCE_THRESHOLD = 0.5
        
        if distance <= DISTANCE_THRESHOLD:
            print(json.dumps({
                "status": "success", 
                "type": "semantic", 
                "distance": round(distance, 4), # Optional: for debugging
                "citation": metadata['path'], 
                "image_path": best_match_id
            }))
        else:
            # Distance is too high; meaning it's a weak/unrelated guess
            print(json.dumps({"status": "error", "message": f"No closely related images found in the database."}))
    else:
        print(json.dumps({"status": "error", "message": "No matching images found in the database."}))

def main():
    parser = argparse.ArgumentParser(description="CLI Image Retrieval Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Scan folders")
    ingest_parser.add_argument("path", type=str)

    query_parser = subparsers.add_parser("search", help="Run query")
    query_parser.add_argument("text", type=str)

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args.path)
    elif args.command == "search":
        handle_query(args.text)

if __name__ == "__main__":
    main()
