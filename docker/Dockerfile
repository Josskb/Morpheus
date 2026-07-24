FROM python:3.12-slim

WORKDIR /app

# Dépendances système (pour ccxt, PostgreSQL client et certains packages ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p logs models

# Port FastAPI (optionnel pour la v1, sera utilisé pour Module 6)
EXPOSE 8000

# Health check via database connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import asyncio; from src.core.database import init_db; asyncio.run(init_db())" || exit 1

CMD ["python", "main.py"]
