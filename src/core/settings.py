"""
core/settings.py
─────────────────
Configuration centralisée via Pydantic Settings.
Toutes les valeurs sont lues depuis .env avec fallback sur les défauts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (deux niveaux au-dessus de ce fichier)
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    tz: str = "Europe/Paris"

    # ── Database ──────────────────────────────────────────────────
    database_url: str = f"sqlite+aiosqlite:///{DATA_DIR}/radar.db"

    # ── Twitter ───────────────────────────────────────────────────
    twitter_bearer_token: str = ""
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""
    twitter_tier: Literal["free", "basic", "pro"] = "free"

    # ── Market Data ───────────────────────────────────────────────
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # ── Telegram ──────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Alertes ───────────────────────────────────────────────────
    min_confidence_score: float = Field(default=0.65, ge=0.0, le=1.0)
    alert_cooldown_minutes: int = Field(default=5, ge=1)

    # ── Récap Telegram ────────────────────────────────────────────
    digest_interval_minutes: int = Field(default=30, ge=5)
    daily_recap_hour_utc: int = Field(default=6, ge=0, le=23)
    alert_recap_interval_hours: int = Field(default=2, ge=1)

    # ── ML ────────────────────────────────────────────────────────
    ml_evaluation_window: Literal["1h", "4h", "24h", "7d"] = "24h"
    ml_min_training_samples: int = Field(default=50, ge=1)
    ml_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Paths (non-env, calculés) ─────────────────────────────────
    @property
    def accounts_config(self) -> Path:
        return CONFIG_DIR / "accounts.yaml"

    @property
    def symbols_config(self) -> Path:
        return CONFIG_DIR / "symbols.yaml"

    @property
    def markets_config(self) -> Path:
        return CONFIG_DIR / "markets.yaml"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    @property
    def twitter_configured(self) -> bool:
        return bool(self.twitter_bearer_token)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance singleton des settings (cachée)."""
    return Settings()
