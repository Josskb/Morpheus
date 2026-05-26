"""
collector/config_loader.py
──────────────────────────
Chargement et validation des fichiers de configuration YAML.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from ..core.settings import get_settings
from .models import AccountsFileConfig


def load_accounts_config(path: Path | None = None) -> AccountsFileConfig:
    """
    Charge et valide accounts.yaml.
    Retourne uniquement les comptes enabled=True.
    """
    if path is None:
        path = get_settings().accounts_config

    if not path.exists():
        logger.warning("accounts.yaml introuvable à {}. Config vide.", path)
        return AccountsFileConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = AccountsFileConfig.model_validate(raw or {})

    enabled = [a for a in config.accounts if a.enabled]
    disabled = len(config.accounts) - len(enabled)

    logger.info(
        "Config chargée : {} comptes actifs, {} désactivés",
        len(enabled),
        disabled,
    )
    return config


def get_enabled_usernames(config: AccountsFileConfig | None = None) -> list[str]:
    """Retourne la liste des usernames actifs triés par priorité."""
    if config is None:
        config = load_accounts_config()

    priority_order = {"high": 0, "medium": 1, "low": 2}
    active = [a for a in config.accounts if a.enabled]
    active.sort(key=lambda a: priority_order.get(a.priority, 99))
    return [a.username for a in active]
