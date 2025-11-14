# src/config.py
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Modèles
SENTENCE_MODEL = "all-MiniLM-L6-v2"  # Rapide et léger
EMBEDDING_DIM = 384

# Index
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "metadata.json"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
