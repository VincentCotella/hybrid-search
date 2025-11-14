# src/utils.py
import nltk
from nltk.corpus import stopwords
import re
import unicodedata

# liste des stopwords français
try:
    stopwords.words("french")
except LookupError:
    print("Téléchargement des stopwords NLTK...")
    nltk.download("stopwords")
    print("Téléchargement terminé.")

FRENCH_STOPWORDS = set(stopwords.words("french"))


def clean_and_tokenize(text: str) -> list[str]:
    """
    Nettoie le texte, le tokenise et retire les stopwords.
    - Normalise les caractères Unicode (ex: é -> e, ï -> i)
    - Conversion en minuscules
    - Suppression de la ponctuation et des nombres
    - Tokenisation
    - Suppression des stopwords et des tokens trop courts
    """
    # 1. Normaliser les caractères
    # NFKD sépare les caractères de leur diacritique (ex: 'é' -> 'e' + '´')
    text = unicodedata.normalize("NFKD", text)

    # 2. Garder uniquement les lettres de base et les espaces
    text = re.sub(r"[^a-z\s]", "", text, flags=re.IGNORECASE)

    # 3. Conversion en minuscules et tokenisation
    tokens = text.lower().split()

    # 4. Filtrer les stopwords et les tokens courts
    meaningful_tokens = [
        token for token in tokens if token not in FRENCH_STOPWORDS and len(token) > 2
    ]

    return meaningful_tokens


# fonction pour nettoyer le texte pour l'affichage (sans tokenisation)
def clean_text(text: str) -> str:
    """Nettoie le texte pour l'affichage, sans retirer les stopwords."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z\s]", "", text, flags=re.IGNORECASE)
    return " ".join(text.lower().split())
