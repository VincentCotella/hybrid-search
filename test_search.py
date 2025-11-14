# test_search.py (version avec debug)
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from indexing.bm25_index import load_bm25_index, create_bm25_index
from indexing.faiss_index import load_faiss_index
from sentence_transformers import SentenceTransformer
from src.utils import clean_and_tokenize
import numpy as np


def test_search():
    # Charger les indexes
    try:
        bm25 = load_bm25_index()
    except FileNotFoundError:
        print("⚠️ Index BM25 non trouvé, recréation...")
        create_bm25_index()
        bm25 = load_bm25_index()

    faiss_index, metadata = load_faiss_index()
    model = SentenceTransformer(metadata["model_name"])

    # Requête de test
    query = "réseaux de neurones"
    tokenized_query = clean_and_tokenize(query)

    print(f"🔍 Requête : '{query}'")
    print(f"🔤 Tokens de la requête : {tokenized_query}\n")

    # --- DEBUG BM25 ---
    print("📊 Analyse BM25 :")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_results = np.argsort(bm25_scores)[::-1][:3]

    for i, idx in enumerate(bm25_results):
        doc = metadata["documents"][idx]
        doc_tokens = doc["content"].split()
        common_tokens = set(tokenized_query) & set(doc_tokens)
        score = bm25_scores[idx]
        print(f"  {i+1}. {doc['title']}")
        print(f"     Score BM25 : {score:.2f}")
        print(f"     Tokens en commun : {common_tokens}\n")

    # --- Recherche Sémantique ---
    print("🧠 Résultats Sémantique:")
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(
        query_embedding, axis=1, keepdims=True
    )
    faiss_scores, faiss_results = faiss_index.search(query_embedding, 3)

    for i, (idx, score) in enumerate(zip(faiss_results[0], faiss_scores[0])):
        doc = metadata["documents"][idx]
        print(f"  {i+1}. {doc['title']} (score: {score:.4f})")


if __name__ == "__main__":
    test_search()
