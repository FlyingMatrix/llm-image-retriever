import os
import json
import chromadb
from PIL import Image
import ollama
from pydantic import BaseModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

class SearchParameters(BaseModel):
    search_term: str
    project_filter: str | None

# 1. Initialize ChromaDB with its native ImageLoader
client = chromadb.PersistentClient(path="./chroma_db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

# The collection needs the data_loader specified to handle image paths natively
collection = client.get_or_create_collection(
    name="project_images", 
    embedding_function=embedding_function,
    data_loader=image_loader
)

# 2. Optimized LLM Intent Parser
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

# 3. Robust Ingestion Script
def index_images(base_folder: str):
    """Indexes local files properly using Chroma's URIs tracking."""
    supported = ('.png', '.jpg', '.jpeg', '.webp')
    if not os.path.exists(base_folder):
        print(f"Directory {base_folder} not found. Please create it first.")
        return

    ids, uris, metadatas = [], [], []
    
    for project_name in os.listdir(base_folder):
        project_path = os.path.join(base_folder, project_name)
        if os.path.isdir(project_path):
            for file_name in os.listdir(project_path):
                if file_name.lower().endswith(supported):
                    file_path = os.path.abspath(os.path.join(project_path, file_name))
                    
                    ids.append(file_path)
                    uris.append(file_path) # Direct path to image file
                    metadatas.append({"project": project_name, "path": file_path})

    if ids:
        # ChromaDB uses uris + data_loader to calculate visual vector spaces seamlessly
        collection.add(ids=ids, uris=uris, metadatas=metadatas)
        print(f"Indexed {len(ids)} images successfully.")

# 4. Retrieval Script
def dynamic_retrieve(human_query: str, base_folder: str):
    """Parses natural sentence and searches database with case-safe constraints."""
    print(f"\nUser Query: '{human_query}'")
    
    # Get local project list dynamically (e.g., ['biotronik', 'skon', 'smartray_methology'])
    valid_projects = []
    if os.path.exists(base_folder):
        valid_projects = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # Run LLM Intent Extraction
    parsed_args = parse_query_with_llm(human_query, valid_projects)
    print(f"-> Extracted Search Term: '{parsed_args.search_term}'")
    print(f"-> Extracted Project Filter: '{parsed_args.project_filter}'")
    
    # Enforce case-safe string evaluation against database
    where_clause = {"project": parsed_args.project_filter} if parsed_args.project_filter else None
    
    results = collection.query(
        query_texts=[parsed_args.search_term],
        n_results=1,
        where=where_clause
    )
    
    if results['ids'] and results['ids'][0]:
        best_match_id = results['ids'][0][0]
        metadata = results['metadatas'][0][0]
        
        print(f"\n[SUCCESS] Match Found!")
        print(f"Citation Path: {metadata['path']}\n")
        return Image.open(best_match_id), metadata['path']
    else:
        print("\n[WARNING] No matching images found in database setup.\n")
        return None, None

if __name__ == "__main__":
    base_directory = "./projects" 
    
    # Step 1: Re-index your files using the updated structural ingestion logic
    print("Refreshing vector database index...")
    index_images(base_directory)
    
    # Step 2: Test Evaluation
    query = "Show me an image of sample from the project of Biotronik."
    img, path = dynamic_retrieve(query, base_directory)
    
    if img:
        img.show()