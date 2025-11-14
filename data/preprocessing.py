# data/preprocessing.py
import json
import re
from pathlib import Path

def clean_text(text):
    """Nettoyage de base du texte"""
    # Conversion en minuscules
    text = text.lower()
    # Suppression des caractères spéciaux
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_documents():
    """Prétraite les documents et les sauvegarde"""
    data_dir = Path("data")
    
    # Charger documents bruts
    with open(data_dir / "documents.json", "r") as f:
        documents = json.load(f)
    
    # Prétraiter chaque document
    processed_docs = []
    for doc in documents:
        processed_doc = {
            "id": doc["id"],
            "title": doc["title"],
            "content": clean_text(doc["content"]),
            "category": doc["category"],
            "difficulty": doc["difficulty"]
        }
        processed_docs.append(processed_doc)
    
    # Sauvegarder documents prétraités
    with open(data_dir / "processed_documents.json", "w", encoding='utf-8') as f:
        json.dump(processed_docs, f, indent=2)
    
    print(f"✅ {len(processed_docs)} documents prétraités")
    return processed_docs