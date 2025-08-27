import os
import json
import hashlib
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ingestion.parsers import PARSER_MAPPING
from ingestion.chunking import chunking_strategy_A # Or chunking_strategy_B

CORPUS_PATH = "corpus"
DB_PATH = "db"
HASHES_PATH = "hashes.json"
BM25_INDEX_PATH = "bm25_index.pkl"
# This is the model we'll use to create numerical representations of our text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_hashes():
    """Loads a dictionary of previously processed file hashes."""
    if os.path.exists(HASHES_PATH):
        with open(HASHES_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    """Saves the dictionary of file hashes."""
    with open(HASHES_PATH, 'w') as f:
        json.dump(hashes, f, indent=4)

def get_file_hash(file_path):
    """Calculates the SHA256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    print("Starting ingestion process...")
    
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="ai_history")
    
    # Load the embedding model (this will download it on the first run)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    processed_files_hashes = load_hashes()
    all_chunked_docs = []
    files_to_process = []

    # --- 1. Detect changed files ---
    print("Detecting file changes...")
    current_files_hashes = {}
    for root, _, files in os.walk(CORPUS_PATH):
        for file_name in files:
            if file_name.startswith('.'): continue
            file_path = os.path.join(root, file_name)
            file_hash = get_file_hash(file_path)
            current_files_hashes[file_path] = file_hash

            if processed_files_hashes.get(file_path) != file_hash:
                print(f"  - DETECTED CHANGE in {file_name}")
                files_to_process.append(file_path)
    
    if not files_to_process:
        print("No file changes detected. Ingestion is up-to-date.")
        return

    # --- 2. Process and chunk changed files ---
    for file_path in files_to_process:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in PARSER_MAPPING:
            print(f"--> Skipping {file_path} (unsupported)")
            continue
        
        # Delete old chunks from this file if it was processed before
        if file_path in processed_files_hashes:
            print(f"  - Deleting old entries for {os.path.basename(file_path)}...")
            collection.delete(where={"source": os.path.basename(file_path)})

        print(f"--> Parsing and chunking {file_path}...")
        parser_func = PARSER_MAPPING[file_ext]
        text, _ = parser_func(file_path)
        
        # We choose our chunking strategy here
        chunks = chunking_strategy_A(text, source=os.path.basename(file_path))
        all_chunked_docs.extend(chunks)
        print(f"    - Created {len(chunks)} chunks.")

    # --- 3. Generate embeddings and add to ChromaDB ---
    if all_chunked_docs:
        print(f"Generating embeddings for {len(all_chunked_docs)} chunks...")
        
        chunk_texts = [doc['text'] for doc in all_chunked_docs]
        embeddings = model.encode(chunk_texts, show_progress_bar=True).tolist()
        
        print("Adding documents to vector store...")
        collection.add(
            ids=[doc['id'] for doc in all_chunked_docs],
            documents=[doc['text'] for doc in all_chunked_docs],
            metadatas=[doc['metadata'] for doc in all_chunked_docs],
            embeddings=embeddings
        )
    
    # --- 4. Create and save BM25 index ---
    print("Updating lexical (BM25) index...")
    # We need all documents in the collection for a complete BM25 index
    all_docs_in_db = collection.get(include=["documents"])
    tokenized_corpus = [doc.split(" ") for doc in all_docs_in_db['documents']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
        
    # --- 5. Save the new hashes ---
    save_hashes(current_files_hashes)
    print("Ingestion process finished successfully.")

if __name__ == "__main__":
    main()