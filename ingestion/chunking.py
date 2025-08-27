from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def chunking_strategy_A(text: str, source: str) -> list[dict]:
    """
    Strategy A: Simple sliding window using RecursiveCharacterTextSplitter.
    """
    # This splitter tries to keep paragraphs, then sentences, then words together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    
    # Structure the chunks with metadata
    chunked_docs = []
    for i, chunk_text in enumerate(chunks):
        chunked_docs.append({
            "id": f"{source}-{i}",
            "text": chunk_text,
            "metadata": {"source": source, "chunk_num": i}
        })
    return chunked_docs

def chunking_strategy_B(text: str, source: str) -> list[dict]:
    """
    Strategy B: Heading-aware splitting.
    First splits by markdown/HTML headings, then uses recursive splitting on those sections.
    This is a placeholder for a more complex strategy. For now, it's the same as A.
    In a real project, you would add regex splitting for headings here.
    """
    
    return chunking_strategy_A(text, source) # Fallback to Strategy A for simplicity