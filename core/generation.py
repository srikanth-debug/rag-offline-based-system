import os
import requests
import json
from dotenv import load_dotenv

# --- Load Configuration ---
load_dotenv()
# We can still point to the model folder for consistency, but the key is the model name for Ollama
MODEL_PATH = os.getenv("MODEL_PATH", "models/phi-3-safetensors")
OLLAMA_MODEL_NAME = "phi3" # The name of the model we pulled with "ollama run"

# Ollama's local API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")

class Generator:
    def __init__(self):
        # The __init__ is now incredibly simple! Ollama handles all the heavy lifting.
        print("Initializing generator (using Ollama)...")
        # We can add a quick check to see if the model is available in Ollama
        try:
            response = requests.post("http://127.0.0.1:11434/api/tags", timeout=5)
            if OLLAMA_MODEL_NAME not in [m['name'].split(':')[0] for m in response.json().get('models', [])]:
                 print(f"WARNING: Model '{OLLAMA_MODEL_NAME}' not found in Ollama. Please run 'ollama run {OLLAMA_MODEL_NAME}'")
        except requests.exceptions.RequestException:
            print("WARNING: Could not connect to Ollama server. Please ensure it is running.")
        print("Generator initialized.")

    def generate_response(self, query: str, context: list[dict]) -> str:
        context_str = "\n\n---\n\n".join([f"Source: {doc['metadata'].get('source', 'unknown')} (Chunk {doc['metadata'].get('chunk_num', 'N/A')})\nContent: {doc['text']}" for doc in context])

        # We use the same prompt format that Phi-3 expects
        prompt = f"""<|system|>
You are an expert AI assistant. Answer the user's question based ONLY on the provided context documents.
- For each claim you make, you MUST cite the source document like this: [Source: filename.pdf (Chunk 5)].
- If the context does not contain the answer, you MUST state that you cannot answer.
CONTEXT DOCUMENTS:
{context_str}
<|end|><|user|>
{query}
<|end|><|assistant|>
"""
        
        # Create the payload to send to the Ollama server
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "stream": False # We want the full response at once
        }

        try:
            # Make the web request to the local Ollama server
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
            response.raise_for_status()
            
            # The actual answer text is inside the 'response' key of the JSON
            return response.json().get("response", "Error: Could not parse Ollama response.")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return "Error: Could not connect to the Ollama server. Is it running?"