# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Ajouter le répertoire parent au path pour importer nos modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from indexing.bm25_index import load_bm25_index
from indexing.faiss_index import load_faiss_index
from src.utils import clean_and_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# --- Configuration et Chargement des Modèles (une seule fois au démarrage) ---
print("🚀 Démarrage de l'API et chargement des modèles...")
try:
    bm25 = load_bm25_index()
    faiss_index, metadata = load_faiss_index()
    sentence_model = SentenceTransformer(metadata["model_name"])
    documents = metadata["documents"]
    print("✅ Tous les modèles et indexes chargés avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")
    bm25, faiss_index, metadata, sentence_model, documents = None, None, None, None, None

# --- Définition des Modèles de Données Pydantic ---

class SearchResultItem(BaseModel):
    id: int
    title: str
    content: str
    category: str
    difficulty: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="Hybrid Search API",
    description="Une API de recherche hybride combinant BM25 et Sentence-BERT",
    version="1.0.0"
)

# --- Logique de Recherche Hybride ---

def reciprocal_rank_fusion(bm25_results, semantic_results, k=60):
    """
    Fusionne les résultats de BM25 et de la recherche sémantique en utilisant Reciprocal Rank Fusion (RRF).
    """
    # bm25_results et semantic_results sont des listes de tuples (doc_id, score)
    # On ne garde que le rang (l'index)
    bm25_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(bm25_results)}
    semantic_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(semantic_results)}

    fusion_scores = {}
    all_doc_ids = set(bm25_ranks.keys()).union(set(semantic_ranks.keys()))

    for doc_id in all_doc_ids:
        rrf_score = 0
        if doc_id in bm25_ranks:
            rrf_score += 1 / (k + bm25_ranks[doc_id] + 1) # +1 car les rangs commencent à 0
        if doc_id in semantic_ranks:
            rrf_score += 1 / (k + semantic_ranks[doc_id] + 1)
        fusion_scores[doc_id] = rrf_score

    # Trier par score RRF décroissant
    ranked_results = sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_results

def hybrid_search(query: str, k: int = 10):
    """
    Effectue une recherche hybride sur la requête donnée.
    """
    if not all([bm25, faiss_index, sentence_model, documents]):
        raise HTTPException(status_code=503, detail="Les modèles ne sont pas chargés correctement.")

    # 1. Recherche BM25
    tokenized_query = clean_and_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [(idx, bm25_scores[idx]) for idx in top_k_bm25_indices if bm25_scores[idx] > 0]

    # 2. Recherche Sémantique
    query_embedding = sentence_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, k)
    semantic_results = [(idx, score) for idx, score in zip(faiss_indices[0], faiss_scores[0]) if score > 0]

    # 3. Fusion Hybride (RRF)
    fused_results = reciprocal_rank_fusion(bm25_results, semantic_results)

    # 4. Formater les résultats
    final_results = []
    for doc_id, score in fused_results[:k]:
        doc = documents[doc_id]
        final_results.append(SearchResultItem(
            id=doc["id"],
            title=doc["title"],
            content=doc["content"],
            category=doc["category"],
            difficulty=doc["difficulty"],
            score=score
        ))
    
    return final_results

# --- Endpoints de l'API ---

@app.get("/", tags=["General"])
def read_root():
    """Endpoint racine pour vérifier que l'API est en ligne."""
    return {"message": "Hybrid Search API is online!", "docs": "/docs"}

@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search_endpoint(q: str, k: Optional[int] = 10):
    """
    Effectue une recherche hybride sur les documents.
    
    - **q**: La requête de recherche (ex: "réseaux de neurones").
    - **k**: Le nombre de résultats à retourner (défaut: 10).
    """
    if not q:
        raise HTTPException(status_code=400, detail="Le paramètre de requête 'q' ne peut pas être vide.")
    
    results = hybrid_search(q, k)
    return SearchResponse(query=q, results=results)
