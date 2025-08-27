# 🚀 Production RAG System (Ollama Edition)

**Author:** Srikanth Parsa  
**Date:** August 2025  

A **Retrieval-Augmented Generation (RAG) system** built to answer questions on the **History of Artificial Intelligence** — fully offline, local-first, and production-ready.  

---

## 🔑 Features
- **Corpus**: Mixed data (PDF + Markdown + CSV) on AI history.  
- **Hybrid Retrieval**: BM25 + ChromaDB (dense vectors) + Cross-Encoder reranker.  
- **Local LLM**: Uses **Ollama** with `phi3` model for stable offline generation.  
- **Confidence & Abstention**: Calibrated scores with abstain for irrelevant queries.  
- **Full-Stack**: FastAPI backend + Streamlit chat UI.  
- **Evaluation Suite**: Automated quality tests with `evaluate.py`.  

---

## 🏛️ Architecture


---

## ⚡ Quickstart

### 1. Setup
```bash
# Clone repo
git clone https://github.com/srikanth-debug/rag-offline-based-system.git
cd rag-offline-based-system

# Create environment
conda create --prefix ./env python=3.11 -y
conda activate ./env
pip install -r requirements.txt

# Ingest data
python ingest.py

---
2. Run
# Terminal 1 → Start backend
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 → Start frontend
streamlit run ui.py

3. Evaluation
python evaluate.py


📌 Notes
Runs fully offline on your machine.
Uses external drive setup for large models.
First query may be slow (model loads into RAM).