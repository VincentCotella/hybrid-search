# main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import create_sample_ml_docs
from indexing.bm25_index import create_bm25_index
from indexing.faiss_index import create_faiss_index

def main():
    print("Initialisation du projet de recherche hybride...")
    
    # Données d'exemple
    print("Création données d'exemple...")
    create_sample_ml_docs()
    
    # Créer indexes
    print("Création de l'index BM25...")
    create_bm25_index()
    
    print("Création de l'index FAISS...")
    create_faiss_index()
    
    print("Initialisation terminée ! Lancez l'API avec: python api/main.py")

if __name__ == "__main__":
    main()