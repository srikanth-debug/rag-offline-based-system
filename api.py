# --- api.py ---
# The Final, Corrected Version

import time
import re
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.retrieval import HybridRetriever
from core.generation import Generator

# --- Load Configuration ---
load_dotenv()
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# --- Load Models at Startup ---
print("Loading models, this might take a few minutes...")
retriever = HybridRetriever()
generator = Generator()
print("Models loaded successfully.")

# --- Helper Function ---
def parse_llm_output(raw_text: str) -> tuple[str, list[str]]:
    cleaned_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
    citation_pattern = r"\[Source:.*?\]"
    citations = re.findall(citation_pattern, cleaned_text)
    answer_text = re.sub(citation_pattern, "", cleaned_text).strip()
    cleaned_citations = [c.replace("[Source:", "").replace("]", "").strip() for c in citations]
    return answer_text, cleaned_citations

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    timings: dict

# --- FastAPI Application ---
app = FastAPI(title="RAG System API")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    start_time = time.time()
    
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # 1. Retrieval
    retrieval_start_time = time.time()
    retrieved_docs = retriever.retrieve(request.query, top_k=2)
    retrieval_time = time.time() - retrieval_start_time

    # 2. Confidence Scoring
    if not retrieved_docs:
        confidence = 0.0
    else:
        confidence = retrieved_docs[0]['score']
    
    # 3. Abstention Logic
    if confidence < CONFIDENCE_THRESHOLD:
        total_time = time.time() - start_time
        return QueryResponse(
            answer="I'm sorry, my knowledge is limited to the history of Artificial Intelligence. I cannot answer that question based on the provided documents.",
            citations=[],
            confidence=confidence,
            timings={
                "retrieval": f"{retrieval_time:.2f}s",
                "generation": "0.00s",
                "total": f"{total_time:.2f}s"
            }
        )

    # 4. Generation
    generation_start_time = time.time()
    raw_answer = generator.generate_response(request.query, retrieved_docs)
    generation_time = time.time() - generation_start_time
    
    # 5. Post-processing
    clean_answer, citations = parse_llm_output(raw_answer)
    
    total_time = time.time() - start_time

    return QueryResponse(
        answer=clean_answer,
        citations=citations,
        confidence=confidence,
        timings={
            "retrieval": f"{retrieval_time:.2f}s",
            "generation": f"{generation_time:.2f}s",
            "total": f"{total_time:.2f}s"
        }
    )