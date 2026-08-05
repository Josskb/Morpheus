"""
collector/twitter_client.py
────────────────────────────
Client Twitter API v2 via tweepy.
Abstrait derrière une interface pour faciliter les mocks en test
et le futur switch vers filtered_stream (Basic tier).

Free tier : polling uniquement (get_users_tweets)
Basic tier : polling + filtered_stream
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import tweepy
import tweepy.asynchronous
from loguru import logger

from ..core.settings import get_settings
from .models import RawTweet

# ── Interface abstraite ────────────────────────────────────────────────────────

class TwitterClientBase(ABC):
    """Interface que tout client Twitter doit implémenter."""

    @abstractmethod
    async def get_recent_tweets(
        self,
        username: str,
        max_results: int = 10,
        since: datetime | None = None,
    ) -> list[RawTweet]:
        """Retourne les tweets récents d'un utilisateur."""
        ...

    @abstractmethod
    async def resolve_user_id(self, username: str) -> str | None:
        """Retourne le Twitter ID pour un username donné."""
        ...


# ── Implémentation Tweepy v2 ───────────────────────────────────────────────────

class TweepyClient(TwitterClientBase):
    """
    Client réel basé sur tweepy.AsyncClient.
    Rate limits free tier :
      - 500k tweets lus / mois
      - get_users_tweets : 1 req / 15 min par endpoint (app-level)
      → Avec 300s de polling, ~288 req/jour max → conservateur OK.
    """

    # Champs Twitter à récupérer
    TWEET_FIELDS = [
        "created_at", "text", "lang",
        "public_metrics", "referenced_tweets",
    ]
    USER_FIELDS = ["id", "username", "name"]

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.twitter_bearer_token:
            raise ValueError(
                "TWITTER_BEARER_TOKEN manquant. "
                "Configure ton .env ou utilise MockTwitterClient."
            )
        self._client = tweepy.AsyncClient(
            bearer_token=settings.twitter_bearer_token,
            wait_on_rate_limit=True,   # tweepy gère le sleep automatiquement
        )
        self._user_id_cache: dict[str, str] = {}
        logger.info("TweepyClient initialisé (tier={})", settings.twitter_tier)

    async def resolve_user_id(self, username: str) -> str | None:
        if username in self._user_id_cache:
            return self._user_id_cache[username]
        try:
            resp = await self._client.get_user(username=username)
            if resp.data:
                uid = str(resp.data.id)
                self._user_id_cache[username] = uid
                logger.debug("Résolu @{} → id={}", username, uid)
                return uid
        except tweepy.TweepyException as e:
            logger.error("Impossible de résoudre @{} : {}", username, e)
        return None

    async def get_recent_tweets(
        self,
        username: str,
        max_results: int = 10,
        since: datetime | None = None,
    ) -> list[RawTweet]:
        user_id = await self.resolve_user_id(username)
        if not user_id:
            return []

        kwargs: dict = dict(
            id=user_id,
            max_results=min(max_results, 100),  # API max = 100
            tweet_fields=self.TWEET_FIELDS,
        )
        if since:
            kwargs["start_time"] = since.isoformat()

        try:
            resp = await self._client.get_users_tweets(**kwargs)
        except tweepy.TweepyException as e:
            logger.error("Erreur fetch tweets @{} : {}", username, e)
            return []

        if not resp.data:
            logger.debug("Aucun tweet récent pour @{}", username)
            return []

        tweets: list[RawTweet] = []
        for t in resp.data:
            metrics = t.public_metrics or {}
            is_rt = bool(
                t.referenced_tweets
                and any(r.type == "retweeted" for r in t.referenced_tweets)
            )
            tweets.append(
                RawTweet(
                    tweet_id=str(t.id),
                    username=username,
                    text=t.text,
                    tweeted_at=t.created_at,
                    lang=t.lang,
                    like_count=metrics.get("like_count", 0),
                    retweet_count=metrics.get("retweet_count", 0),
                    reply_count=metrics.get("reply_count", 0),
                    is_retweet=is_rt,
                )
            )
        logger.debug("Fetché {} tweets pour @{}", len(tweets), username)
        return tweets


# ── Mock client (dev sans credentials) ────────────────────────────────────────

class MockTwitterClient(TwitterClientBase):
    """
    Client factice pour développer sans crédentials Twitter.
    Génère des tweets réalistes à partir de templates.
    Usage : APP_ENV=development + pas de TWITTER_BEARER_TOKEN
    """

    _TEMPLATES = [
        "{ticker} looking very bullish here. Strong support at {price}. Accumulating 🚀",
        "Just closed my {ticker} position. +{gain}% in {hours}h. Next target {target}",
        "Warning ⚠️ {ticker} breaking key support. Watch out for further downside.",
        "${ticker} forming a nice cup & handle on the daily. Price target: ${target}",
        "Thread on why {ticker} is my top pick for Q{quarter} 🧵",
        "{ticker} volume spike on low float. This could move fast. DYOR.",
        "Macro update: Fed still hawkish, risk-off environment. Reducing exposure.",
        "${ticker} earnings beat + raised guidance. Gap up incoming tomorrow.",
    ]

    def __init__(self, seed: int = 42) -> None:
        import random
        self._rng = random.Random(seed)
        logger.warning(
            "MockTwitterClient actif — aucun vrai tweet ne sera récupéré. "
            "Configure TWITTER_BEARER_TOKEN pour le client réel."
        )

    async def resolve_user_id(self, username: str) -> str | None:
        return str(abs(hash(username)) % 10**15)

    async def get_recent_tweets(
        self,
        username: str,
        max_results: int = 5,
        since: datetime | None = None,
    ) -> list[RawTweet]:

        tickers = ["BTC", "ETH", "AAPL", "NVDA", "SOL", "TSLA", "SPY"]
        now = datetime.now(tz=timezone.utc)
        tweets = []

        count = self._rng.randint(1, max_results)
        for i in range(count):
            ticker = self._rng.choice(tickers)
            template = self._rng.choice(self._TEMPLATES)
            text = template.format(
                ticker=ticker,
                price=round(self._rng.uniform(100, 50000), 2),
                gain=round(self._rng.uniform(2, 30), 1),
                hours=self._rng.choice([1, 4, 8, 24]),
                target=round(self._rng.uniform(100, 60000), 2),
                quarter=self._rng.randint(1, 4),
            )
            offset = timedelta(minutes=self._rng.randint(5, 240))
            tweets.append(
                RawTweet(
                    tweet_id=str(self._rng.randint(10**17, 10**18)),
                    username=username,
                    text=text,
                    tweeted_at=now - offset,
                    lang="en",
                    like_count=self._rng.randint(0, 2000),
                    retweet_count=self._rng.randint(0, 500),
                    reply_count=self._rng.randint(0, 100),
                    is_retweet=False,
                )
            )
        return tweets


# ── Factory ────────────────────────────────────────────────────────────────────

def create_twitter_client() -> TwitterClientBase:
    """
    Retourne le bon client selon la configuration :
    - Si TWITTER_BEARER_TOKEN est défini → TweepyClient réel
    - Sinon en dev → MockTwitterClient
    """
    settings = get_settings()
    if settings.twitter_configured:
        return TweepyClient()
    if settings.is_dev:
        logger.info("Pas de credentials Twitter → MockTwitterClient activé")
        return MockTwitterClient()
    raise RuntimeError(
        "TWITTER_BEARER_TOKEN requis en production. Configure ton .env."
    )
