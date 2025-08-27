import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from core.retrieval import HybridRetriever
from core.generation import Generator
import time


def main():
    # Initialize the components
    retriever = HybridRetriever()
    generator = Generator()

    print("\n--- AI History RAG System ---")
    print("Ask a question about the history of AI. Type 'exit' to quit.")

    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            break
        
        # 1. Retrieval
        start_time = time.time()
        retrieved_docs = retriever.retrieve(query, top_k=3)
        retrieval_time = time.time() - start_time
        
        if not retrieved_docs:
            print("Could not retrieve any relevant documents.")
            continue

        print(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.")
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"  {i+1}. Source: {doc['metadata']['source']}, Score: {doc['score']:.4f}")

        # 2. Generation
        start_time = time.time()
        answer = generator.generate_response(query, retrieved_docs)
        generation_time = time.time() - start_time
        
        print(f"\nAnswer (generated in {generation_time:.2f} seconds):")
        print(answer)

if __name__ == "__main__":
    main()