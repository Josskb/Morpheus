"""
main.py
────────
Point d'entrée de financial-radar.
Lance collecteur + scheduler + NLP + alertes Telegram en parallèle.

Usage :
  python main.py               # démarre tout
  python main.py --poll-once   # un seul round de polling (test/debug)
  python main.py --init-db     # crée les tables uniquement
  python main.py --nlp-once    # traite les tweets NLP en attente puis quitter
  python main.py --alert-once  # envoie les alertes en attente puis quitter
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

# Setup logging en premier
from src.core.logging import setup_logging

setup_logging()

from src.core.database import init_db
from src.collector.service import CollectorService
from src.market.snapshot_scheduler import SnapshotScheduler
from src.nlp.processor import NLPProcessorService
from src.alerts.service import AlertService


async def run_all() -> None:
    """Lance tous les services en parallèle."""
    await init_db()
    logger.info("Base de données initialisée.")

    scheduler = SnapshotScheduler()
    collector = CollectorService()
    nlp = NLPProcessorService(scheduler=scheduler)
    alerts = AlertService()

    await scheduler.reload_pending_from_db()

    try:
        await asyncio.gather(
            collector.run(),
            scheduler.run(),
            nlp.run(),
            alerts.run(),
        )
    except KeyboardInterrupt:
        logger.info("Arrêt demandé (Ctrl+C).")
    finally:
        collector.stop()
        scheduler.stop()
        nlp.stop()
        await alerts.stop()
        from src.market.fetcher import get_market_fetcher
        await get_market_fetcher().close()
        logger.info("financial-radar arrêté proprement.")


async def poll_once() -> None:
    """Un seul tour de polling pour tester."""
    await init_db()
    collector = CollectorService()
    await collector.sync_accounts_to_db()
    results = await collector.poll_once()
    print(f"\n{'='*50}")
    print(f"Résultats du polling :")
    for username, count in results.items():
        print(f"  @{username:30s} → {count:3d} nouveaux tweets")
    total = sum(results.values())
    print(f"{'='*50}")
    print(f"  TOTAL : {total} tweets\n")


async def nlp_once() -> None:
    """Traite les tweets en attente de NLP puis quitte."""
    await init_db()
    scheduler = SnapshotScheduler()
    nlp = NLPProcessorService(scheduler=scheduler)
    count = await nlp.process_pending()
    print(f"\nNLP : {count} tweets traités.")


async def alert_once() -> None:
    """Envoie les alertes en attente puis quitte."""
    await init_db()
    alerts = AlertService()
    await alerts._bot.start()
    count = await alerts.check_and_send()
    await alerts._bot.stop()
    print(f"\nAlertes : {count} envoyée(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="financial-radar — Aide à la décision financière")
    parser.add_argument("--poll-once", action="store_true", help="Un seul polling puis quitter")
    parser.add_argument("--init-db", action="store_true", help="Initialiser la DB puis quitter")
    parser.add_argument("--nlp-once", action="store_true", help="Traiter les tweets NLP en attente puis quitter")
    parser.add_argument("--alert-once", action="store_true", help="Envoyer les alertes en attente puis quitter")
    args = parser.parse_args()

    if args.init_db:
        asyncio.run(init_db())
        logger.info("DB initialisée.")
        sys.exit(0)

    if args.poll_once:
        asyncio.run(poll_once())
        sys.exit(0)

    if args.nlp_once:
        asyncio.run(nlp_once())
        sys.exit(0)

    if args.alert_once:
        asyncio.run(alert_once())
        sys.exit(0)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
