import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Paths
CHROMA_DB_PATH = "./chroma_db"
DATA_DIR = "./data"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

# Models — all local, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace local
LLM_MODEL = "llama3.2"                  # Ollama local

# ChromaDB
CHROMA_COLLECTION = "niti_aayog_reports"