import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="Moteur de Recherche Hybride", page_icon="🔍", layout="wide"
)

# Titre de l'application
st.title("🔍 Moteur de Recherche Hybride pour Documentation ML")
st.markdown(
    "Recherchez dans la documentation de frameworks ML en combinant recherche lexicale et sémantique."
)

# Initialisation de l'état de la session
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# Configuration de l'URL de l'API
with st.sidebar:
    st.header("Configuration")
    st.session_state.api_url = st.text_input(
        "URL de l'API", value=st.session_state.api_url
    )

    # Récupérer les catégories disponibles
    try:
        response = requests.get(f"{st.session_state.api_url}/categories")
        if response.status_code == 200:
            categories = response.json()["categories"]
        else:
            categories = []
            st.error("Impossible de récupérer les catégories")
    except:
        categories = []
        st.error("Impossible de se connecter à l'API")

# Barre de recherche
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Entrez votre requête :", placeholder="ex: réseaux de neurones"
    )
with col2:
    k = st.number_input("Nombre de résultats", min_value=1, max_value=20, value=5)

# Options de recherche
with st.expander("Options de recherche avancées"):
    search_category = st.selectbox(
        "Filtrer par catégorie (optionnel)", ["Toutes"] + categories
    )

# Bouton de recherche
search_button = st.button("Rechercher")

# Affichage des résultats
if search_button or query:
    if not query:
        st.warning("Veuillez entrer une requête de recherche.")
    else:
        with st.spinner("Recherche en cours..."):
            try:
                # Recherche avec ou sans filtre de catégorie
                if search_category != "Toutes":
                    response = requests.get(
                        f"{st.session_state.api_url}/search/category/{search_category}",
                        params={"q": query, "k": k},
                    )
                else:
                    response = requests.get(
                        f"{st.session_state.api_url}/search",
                        params={"q": query, "k": k},
                    )

                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]

                    st.subheader(
                        f"Résultats pour '{data['query']}' ({len(results)} résultats)"
                    )

                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"### {i+1}. {result['title']}")
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.markdown(f"**Catégorie:** {result['category']}")
                            with col2:
                                st.markdown(f"**Difficulté:** {result['difficulty']}")
                            with col3:
                                st.markdown(f"**Score:** {result['score']:.4f}")

                            st.markdown(f"**Contenu:** {result['content']}")
                            st.divider()
                else:
                    st.error(f"Erreur lors de la recherche: {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Erreur de connexion à l'API: {str(e)}")
