"""
collector/models.py
────────────────────
Schémas Pydantic pour les données brutes du collecteur.
Découple la représentation interne de l'API Twitter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RawTweet(BaseModel):
    """Représentation normalisée d'un tweet, indépendante de la source."""

    tweet_id: str
    username: str
    text: str
    tweeted_at: datetime
    lang: Optional[str] = None
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    is_retweet: bool = False

    # Renseignés directement par certaines sources (ex: StockTwits) —
    # évite de repasser par l'extraction lexicale du NLP quand la source
    # fournit déjà un signal plus fiable (sentiment déclaré par l'auteur).
    tickers: list[str] = Field(default_factory=list)
    sentiment: Optional[float] = None    # -1.0 à 1.0, si fourni par la source
    confidence: Optional[float] = None   # 0 à 1, si fourni par la source

    class Config:
        frozen = True  # immutable après création


class AccountConfig(BaseModel):
    """Représente un compte tel que défini dans accounts.yaml."""

    username: str
    display_name: Optional[str] = None
    markets: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    priority: str = "medium"
    enabled: bool = True


class PollingConfig(BaseModel):
    interval_seconds: int = 300
    tweets_per_account: int = 5
    lookback_hours: int = 24


class FiltersConfig(BaseModel):
    exclude_retweets: bool = True
    exclude_replies: bool = False
    min_length: int = 20
    languages: list[str] = Field(default_factory=lambda: ["en", "fr"])


class AccountsFileConfig(BaseModel):
    """Schéma complet du fichier accounts.yaml."""
    accounts: list[AccountConfig] = Field(default_factory=list)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
