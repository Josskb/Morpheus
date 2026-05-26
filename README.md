# 📡 financial-radar

Système personnel d'aide à la décision financière basé sur l'analyse de tweets de traders/analystes.

## Architecture

```
financial-radar/
├── src/
│   ├── core/           # DB, settings, logging
│   ├── collector/      # Twitter polling + stockage
│   ├── market/         # Prix stocks (yfinance) + crypto (ccxt)
│   ├── nlp/            # Extraction tickers, sentiment (Module 3 — à venir)
│   ├── ml/             # Scoring fiabilité (Module 4 — à venir)
│   ├── alerts/         # Bot Telegram (Module 5 — à venir)
│   └── dashboard/      # FastAPI + React (Module 6 — à venir)
├── config/
│   ├── accounts.yaml   # Comptes Twitter à suivre
│   └── markets.yaml    # Actifs et fenêtres de tracking
├── tests/
├── data/               # SQLite DB + données brutes
├── models/             # Modèles ML sérialisés
└── logs/
```

## Installation rapide

```bash
# 1. Clone et setup
git clone <repo>
cd financial-radar
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# Édite .env avec tes credentials
# Édite config/accounts.yaml avec tes comptes Twitter

# 3. Lancer
python main.py --init-db     # Crée les tables
python main.py --poll-once   # Test un seul polling
python main.py               # Lance en continu
```

## Sans credentials Twitter (mode dev)

Le système démarre avec `MockTwitterClient` automatiquement si `TWITTER_BEARER_TOKEN` est vide.
Des tweets réalistes sont générés pour tester le pipeline NLP/ML.

## Configuration des comptes

Édite `config/accounts.yaml` :
```yaml
accounts:
  - username: ton_trader_prefere
    display_name: "Nom Affiché"
    markets: [crypto]           # crypto | stocks | options
    tags: [ta, bitcoin]
    priority: high              # high | medium | low
    enabled: true
```

## Tests

```bash
pytest                         # Tous les tests
pytest tests/unit/             # Tests unitaires seulement
pytest -k "test_collector"     # Tests spécifiques
```

## Déploiement VPS (Hetzner CAX11 recommandé)

```bash
# Sur ton VPS
docker compose up -d
docker compose logs -f radar
```

## Modules à venir

- **Module 3 — NLP** : FinBERT + extraction tickers/prix cibles
- **Module 4 — ML** : XGBoost + scoring fiabilité par compte
- **Module 5 — Alertes** : Bot Telegram avec scores de confiance
- **Module 6 — Dashboard** : React PWA consultable sur mobile

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Runtime | Python 3.11+ |
| DB | SQLite (dev) → PostgreSQL (prod) |
| ORM | SQLAlchemy 2.0 async |
| Twitter | tweepy v4 |
| Stocks | yfinance |
| Crypto | ccxt (Binance) |
| API | FastAPI |
| Frontend | React + Tailwind (PWA) |
| Alertes | python-telegram-bot |
| Logs | loguru |
| Deploy | Docker Compose |
