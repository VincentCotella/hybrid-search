# indexing/faiss_index.py
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.config import FAISS_INDEX_PATH, SENTENCE_MODEL, DATA_DIR


def create_faiss_index():
    """Crée un index FAISS à partir des embeddings des documents"""
    # Charger documents prétraités
    with open(DATA_DIR / "documents.json", "r") as f:
        documents = json.load(f)

    # Charger modèle de phrases
    model = SentenceTransformer(SENTENCE_MODEL)

    # Préparer les textes (titre + contenu)
    texts = [f"{doc['title']} {doc['content']}" for doc in documents]

    # Générer embeddings
    print("Génération des embeddings")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Normaliser les embeddings (pour cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Créer index FAISS
    dimension = embeddings.shape[1]  # 384 pour MiniLM
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine si normalisés

    # Ajouter les embeddings à l'index
    index.add(embeddings)

    # Sauvegarder l'index
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Sauvegarder les métadonnées
    metadata = {
        "documents": documents,
        "model_name": SENTENCE_MODEL,
        "dimension": dimension,
    }
    with open(DATA_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index FAISS créé avec {len(documents)} documents")
    return index


def load_faiss_index():
    """Charge l'index FAISS existant"""
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
