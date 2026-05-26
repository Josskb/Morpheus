FROM python:3.11-slim

WORKDIR /app

# Dépendances système (pour ccxt et certains packages ML)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p data logs models

# Port FastAPI
EXPOSE 8000

CMD ["python", "main.py"]
