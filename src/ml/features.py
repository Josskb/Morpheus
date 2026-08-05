"""
ml/features.py
────────────────
Construction du vecteur de features ML à partir d'un tweet enrichi (NLP),
de son compte, et du snapshot marché au moment du tweet (t=0).

Pur et sans dépendance DB/IO — partagé entre scorer.py (inférence) et
trainer.py (entraînement) pour éviter toute duplication de logique.

Important : n'utilise jamais price_1h/4h/24h/7d ni change_1h/4h/24h/7d —
ces champs ne sont connus qu'après coup et servent uniquement au labeling
(voir labeler.py). Les inclure ici introduirait une fuite de données.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

FEATURE_NAMES: list[str] = [
    "sentiment",
    "confidence",
    "urgency_score",
    "is_long",
    "is_short",
    "num_tickers",
    "has_target_price",
    "has_stop_loss",
    "log_like_count",
    "log_retweet_count",
    "log_reply_count",
    "is_crypto",
    "log_price_at_tweet",
    "log_volume_at_tweet",
    "account_reliability_score",
    "account_win_rate",
    "account_avg_roi",
    "log_account_total_calls",
]


def primary_ticker(tweet) -> str | None:
    """
    Un tweet peut référencer plusieurs tickers, mais le scoring est par tweet.
    Simplification MVP (identique à celle déjà faite par le NLP pour call_type) :
    le premier ticker extrait est le ticker "principal" utilisé pour les
    features de prix et pour le labeling.
    """
    if not tweet.tickers:
        return None
    try:
        tickers = json.loads(tweet.tickers)
    except (TypeError, ValueError):
        return None
    return tickers[0] if tickers else None


def get_primary_snapshot(tweet, snapshots: Iterable):
    """Retrouve le MarketSnapshot correspondant au ticker principal du tweet."""
    ticker = primary_ticker(tweet)
    if ticker is None:
        return None
    for snap in snapshots:
        if snap.ticker == ticker:
            return snap
    return None


def build_features(tweet, account, snapshot) -> dict[str, float]:
    """
    Construit le vecteur de features pour un tweet donné.

    `snapshot` est le MarketSnapshot principal (t=0) associé au tweet,
    ou None s'il n'existe pas encore.
    """
    try:
        num_tickers = len(json.loads(tweet.tickers)) if tweet.tickers else 0
    except (TypeError, ValueError):
        num_tickers = 0

    return {
        "sentiment": tweet.sentiment or 0.0,
        "confidence": tweet.confidence or 0.0,
        "urgency_score": tweet.urgency_score or 0.0,
        "is_long": 1.0 if tweet.call_type == "long" else 0.0,
        "is_short": 1.0 if tweet.call_type == "short" else 0.0,
        "num_tickers": float(num_tickers),
        "has_target_price": 1.0 if tweet.target_price is not None else 0.0,
        "has_stop_loss": 1.0 if tweet.stop_loss is not None else 0.0,
        "log_like_count": math.log1p(tweet.like_count or 0),
        "log_retweet_count": math.log1p(tweet.retweet_count or 0),
        "log_reply_count": math.log1p(tweet.reply_count or 0),
        "is_crypto": 1.0 if snapshot is not None and snapshot.market_type == "crypto" else 0.0,
        "log_price_at_tweet": math.log1p(snapshot.price_at_tweet) if snapshot and snapshot.price_at_tweet else 0.0,
        "log_volume_at_tweet": math.log1p(snapshot.volume_at_tweet) if snapshot and snapshot.volume_at_tweet else 0.0,
        "account_reliability_score": account.reliability_score if account.reliability_score is not None else 0.5,
        "account_win_rate": account.win_rate if account.win_rate is not None else 0.5,
        "account_avg_roi": account.avg_roi if account.avg_roi is not None else 0.0,
        "log_account_total_calls": math.log1p(account.total_calls or 0),
    }


def to_vector(features: dict[str, float]) -> list[float]:
    """Convertit le dict de features en vecteur ordonné selon FEATURE_NAMES."""
    return [features[name] for name in FEATURE_NAMES]
