"""
main.py
────────
Point d'entrée de financial-radar.
Lance le collecteur + le scheduler de snapshots en parallèle.

Usage :
  python main.py               # démarre tout
  python main.py --poll-once   # un seul round de polling (test/debug)
  python main.py --init-db     # crée les tables uniquement
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


async def run_all() -> None:
    """Lance tous les services en parallèle."""
    await init_db()
    logger.info("Base de données initialisée.")

    collector = CollectorService()
    scheduler = SnapshotScheduler()

    # Recharge les snapshots manquants depuis la DB au démarrage
    await scheduler.reload_pending_from_db()

    try:
        await asyncio.gather(
            collector.run(),
            scheduler.run(),
        )
    except KeyboardInterrupt:
        logger.info("Arrêt demandé (Ctrl+C).")
    finally:
        collector.stop()
        scheduler.stop()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="financial-radar — Aide à la décision financière")
    parser.add_argument("--poll-once", action="store_true", help="Un seul polling puis quitter")
    parser.add_argument("--init-db", action="store_true", help="Initialiser la DB puis quitter")
    args = parser.parse_args()

    if args.init_db:
        asyncio.run(init_db())
        logger.info("DB initialisée.")
        sys.exit(0)

    if args.poll_once:
        asyncio.run(poll_once())
        sys.exit(0)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
