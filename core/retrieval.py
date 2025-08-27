import pickle
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- Constants ---
DB_PATH = "db"
BM25_INDEX_PATH = "bm25_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class HybridRetriever:
    def __init__(self):
        print("Initializing retriever (optimized hybrid mode)...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name="ai_history")
        
        with open(BM25_INDEX_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        # --- THE FIX: Load only documents and metadata into memory ---
        # This is small and memory-safe but gives us fast lookups.
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.bm25_doc_texts = all_data['documents']
        self.doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
        print("Retriever initialized.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # 1. Dense Search
        query_embedding = self.embed_model.encode(query).tolist()
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        # 2. Lexical Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        # 3. Combine Results
        combined_docs_map = {}
        # Add dense results
        for doc_text, metadata in zip(dense_results['documents'][0], dense_results['metadatas'][0]):
            combined_docs_map[doc_text] = metadata
        
        # Add lexical results using our fast in-memory metadata lookup
        for i in top_n_indices:
            doc_text = self.bm25_doc_texts[i]
            if doc_text not in combined_docs_map:
                combined_docs_map[doc_text] = self.doc_to_meta.get(doc_text, {"source": "unknown"})

        # 4. Reranking
        doc_meta_list = list(combined_docs_map.items())
        if not doc_meta_list: return []
            
        rerank_pairs = [[query, doc_text] for doc_text, _ in doc_meta_list]
        raw_scores = self.reranker.predict(rerank_pairs)
        
        # Normalize scores
        normalized_scores = 1 / (1 + np.exp(-np.array(raw_scores)))
        
        reranked_results = []
        for i, (doc_text, metadata) in enumerate(doc_meta_list):
            reranked_results.append({
                "text": doc_text,
                "score": normalized_scores[i],
                "metadata": metadata
            })

        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        return reranked_results[:top_k]