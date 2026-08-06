"""
collector/stocktwits_client.py
───────────────────────────────
Client pour l'API publique StockTwits (gratuite, sans clé).

Récupère le flux de messages pour un symbole donné
(actions : "AAPL" ; crypto : "BTC.X", "ETH.X", ...).

Beaucoup de messages incluent un sentiment déclaré par l'auteur
(Bullish/Bearish) → utilisé directement, sans passer par l'extraction
lexicale du NLP local (voir RawTweet.sentiment/confidence/tickers).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import httpx
from loguru import logger

from .models import RawTweet
from .twitter_client import TwitterClientBase

BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"

# Sentiment déclaré par l'auteur → score numérique
SENTIMENT_SCORES = {
    "Bullish": 0.6,
    "Bearish": -0.6,
}


class StockTwitsClient(TwitterClientBase):
    """Client pour le flux StockTwits d'un symbole."""

    async def resolve_user_id(self, username: str) -> str | None:
        # StockTwits identifie par symbole, pas par ID utilisateur — pas
        # de résolution nécessaire (uniquement utilisé en interne par
        # TweepyClient, jamais appelé depuis CollectorService).
        return None

    async def get_recent_tweets(
        self,
        username: str,
        max_results: int = 10,
        since: datetime | None = None,
    ) -> list[RawTweet]:
        """
        `username` est ici un symbole StockTwits (ex: "AAPL", "BTC.X").
        """
        symbol = username
        url = f"{BASE_URL}/{symbol}.json"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
        except Exception as e:
            logger.error("Erreur fetch StockTwits {} : {}", symbol, e)
            return []

        data = resp.json()
        messages = data.get("messages", [])

        tweets: list[RawTweet] = []
        for msg in messages[:max_results]:
            try:
                created_at = datetime.fromisoformat(
                    msg["created_at"].replace("Z", "+00:00")
                )
            except Exception:
                created_at = datetime.now(tz=timezone.utc)

            if since and created_at < since:
                continue

            sentiment_label = (
                msg.get("entities", {}).get("sentiment", {}) or {}
            ).get("basic")
            sentiment = SENTIMENT_SCORES.get(sentiment_label)
            confidence = abs(sentiment) if sentiment is not None else None

            author = msg.get("user", {}).get("username", "?")
            body = html.unescape(msg.get("body", "")).strip()
            text = f"(@{author}) {body}"

            tickers = [s["symbol"] for s in msg.get("symbols", [])]

            tweets.append(RawTweet(
                tweet_id=f"st_{msg['id']}",
                username=author,
                text=text,
                tweeted_at=created_at,
                lang=None,
                like_count=msg.get("likes", {}).get("total", 0) if msg.get("likes") else 0,
                retweet_count=0,
                reply_count=0,
                is_retweet=False,
                tickers=tickers,
                sentiment=sentiment,
                confidence=confidence,
            ))

        logger.debug("Fetché {} messages StockTwits pour {}", len(tweets), symbol)
        return tweets
