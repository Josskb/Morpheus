"""
collector/yfinance_news_client.py
───────────────────────────────────
Collecte de signaux via les flux de news yfinance.
Utile pour les titres peu couverts par StockTwits (petites caps, marchés
étrangers). Chaque titre d'article est traité comme un "tweet" et passe
par le pipeline NLP habituel.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import yfinance as yf
from loguru import logger

from .models import RawTweet
from .twitter_client import TwitterClientBase


class YFinanceNewsClient(TwitterClientBase):
    """Fetche les news yfinance et les convertit en RawTweet."""

    async def resolve_user_id(self, username: str) -> str | None:
        # Pas de notion d'ID utilisateur pour un flux de news par ticker —
        # uniquement utilisé en interne par TweepyClient, jamais appelé
        # depuis CollectorService.
        return None

    async def get_recent_tweets(
        self,
        username: str,
        max_results: int = 20,
        since: datetime | None = None,
    ) -> list[RawTweet]:
        try:
            ticker_obj = yf.Ticker(username)
            news = await asyncio.to_thread(lambda: ticker_obj.news)
        except Exception as e:
            logger.warning("yfinance news {} : {}", username, e)
            return []

        results: list[RawTweet] = []
        for article in (news or [])[:max_results]:
            title = (
                article.get("title")
                or (article.get("content") or {}).get("title", "")
            )
            if not title:
                continue

            pub = article.get("providerPublishTime") or article.get("pubDate")
            if isinstance(pub, (int, float)):
                tweeted_at = datetime.fromtimestamp(pub, tz=timezone.utc)
            else:
                tweeted_at = datetime.now(tz=timezone.utc)

            if since and tweeted_at < since:
                continue

            uid = (
                article.get("uuid")
                or article.get("id")
                or f"{username}_{int(tweeted_at.timestamp())}"
            )

            results.append(RawTweet(
                tweet_id=f"yn_{uid}",
                username=username,
                text=title,
                lang="en",
                tweeted_at=tweeted_at,
                tickers=[username],
                sentiment=None,
                confidence=None,
            ))

        logger.debug("yfinance news {} : {} articles récupérés", username, len(results))
        return results
