
# Construction de l'environnement
FROM python:3.9-slim AS builder

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir --user -r requirements.txt

# Image de production finale
FROM python:3.9-slim

WORKDIR /app

# Copier les dépendances installées depuis l'étape de construction
COPY --from=builder /root/.local /root/.local

# Copier le code de l'application
COPY . .

# Mettre à jour le PATH pour utiliser les packages installés par l'utilisateur
ENV PATH=/root/.local/bin:$PATH

# Exposer le port de l'API
EXPOSE 8000

# Commande pour lancer l'API avec Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]