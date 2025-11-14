# indexing/bm25_index.py
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.config import BM25_INDEX_PATH, DATA_DIR
from src.utils import clean_and_tokenize
import json


def create_bm25_index():
    """Crée un index BM25 à partir des documents prétraités (titre + contenu)"""
    with open(DATA_DIR / "documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    print("\n--- DÉBUT DU DEBUG DE L'INDEXATION BM25 (TITRE + CONTENU) ---")
    tokenized_docs = []
    for i, doc in enumerate(documents):
        # On combine le titre et le contenu avant de nettoyer
        text_to_index = f"{doc['title']} {doc['content']}"
        tokens = clean_and_tokenize(text_to_index)

        print(f"Doc {i+1}: '{doc['title']}'")
        print(f"  -> Tokens nettoyés (titre+contenu) : {tokens}")
        tokenized_docs.append(tokens)
    print("--- FIN DU DEBUG DE L'INDEXATION BM25 ---\n")

    bm25 = BM25Okapi(tokenized_docs)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(
        f"✅ Index BM25 créé avec {len(documents)} documents (basés sur titre+contenu)"
    )
    return bm25


def load_bm25_index():
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)
