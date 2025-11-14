# data/loader.py
import json
from pathlib import Path


def create_sample_ml_docs():
    """Crée un jeu de données d'exemple sur le Machine Learning."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    documents = [
        {
            "id": 1,
            "title": "Introduction à la régression linéaire",
            "content": "La régression linéaire est une méthode statistique qui permet de modéliser la relation entre une variable dépendante et une ou plusieurs variables indépendantes en ajustant une droite aux données.",
            "category": "supervised_learning",
            "difficulty": "beginner",
        },
        {
            "id": 2,
            "title": "Implémentation des réseaux de neurones avec PyTorch",
            "content": "PyTorch est une bibliothèque d'apprentissage automatique open source basée sur Torch. Elle est particulièrement utilisée pour les applications de deep learning comme la vision par ordinateur et le traitement du langage naturel.",
            "category": "deep_learning",
            "difficulty": "intermediate",
        },
        {
            "id": 3,
            "title": "Algorithmes de clustering non supervisé",
            "content": "Le clustering est une technique d'apprentissage non supervisé qui regroupe des points de données similaires. Les algorithmes courants incluent K-means, DBSCAN et le clustering hiérarchique.",
            "category": "unsupervised_learning",
            "difficulty": "intermediate",
        },
        {
            "id": 4,
            "title": "Optimisation des hyperparamètres avec GridSearchCV",
            "content": "GridSearchCV est une technique de Scikit-learn pour trouver les hyperparamètres optimaux d'un modèle en évaluant systématiquement toutes les combinaisons possibles via validation croisée.",
            "category": "model_selection",
            "difficulty": "intermediate",
        },
        {
            "id": 5,
            "title": "Les arbres de décision pour la classification",
            "content": "Les arbres de décision sont des modèles de classification et de régression qui construisent un arbre de décisions basé sur les caractéristiques des données. Ils sont faciles à interpréter et à visualiser.",
            "category": "supervised_learning",
            "difficulty": "beginner",
        },
        {
            "id": 6,
            "title": "Réduction de dimensionnalité avec l'ACP (PCA)",
            "content": "L'Analyse en Composantes Principales (ACP) est une technique non supervisée pour réduire la dimensionnalité d'un dataset en transformant les variables en un plus petit nombre de composantes principales.",
            "category": "unsupervised_learning",
            "difficulty": "intermediate",
        },
        {
            "id": 7,
            "title": "Random Forest : un ensemble d'arbres de décision",
            "content": "Random Forest est un algorithme d'apprentissage ensembliste qui construit multiples arbres de décision et les agrège pour améliorer la précision prédictive et contrôler le surapprentissage.",
            "category": "supervised_learning",
            "difficulty": "intermediate",
        },
        {
            "id": 8,
            "title": "Transformer le NLP avec BERT",
            "content": "BERT (Bidirectional Encoder Representations from Transformers) est un modèle de langage pré-entraîné qui a révolutionné le NLP en comprenant le contexte des mots dans une phrase de manière bidirectionnelle.",
            "category": "nlp",
            "difficulty": "advanced",
        },
    ]

    # Sauvegarder les documents
    with open(data_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"{len(documents)} documents exemples créés dans {data_dir}/documents.json")
    return documents
