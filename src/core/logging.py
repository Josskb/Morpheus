"""
core/logging.py
───────────────
Configuration loguru avec rotation, niveaux par module,
et format structuré JSON en production.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .settings import get_settings, LOGS_DIR


def setup_logging() -> None:
    """
    Initialise loguru. À appeler une seule fois au démarrage de l'app.
    - Dev  : logs colorés sur stdout + fichier rotatif
    - Prod : logs JSON sur stdout (pour collecte par journald/loki)
    """
    settings = get_settings()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Supprime le handler par défaut de loguru
    logger.remove()

    if settings.is_dev:
        # ── Console lisible pour le dev ──────────────────────────
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        # ── Fichier rotatif (garde 7 jours) ─────────────────────
        logger.add(
            LOGS_DIR / "radar_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="00:00",    # nouveau fichier à minuit
            retention="7 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            encoding="utf-8",
        )
    else:
        # ── JSON structuré pour la prod (parsing facile) ─────────
        logger.add(
            sys.stdout,
            level=settings.log_level,
            serialize=True,      # émet du JSON pur
        )
        logger.add(
            LOGS_DIR / "radar_{time:YYYY-MM-DD}.log",
            level="WARNING",
            rotation="00:00",
            retention="30 days",
            compression="gz",
            serialize=True,
        )

    logger.info(
        "Logging initialisé",
        env=settings.app_env,
        level=settings.log_level,
    )
