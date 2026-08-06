"""
tests/unit/test_collector.py
─────────────────────────────
Tests unitaires du module collector.
Utilise MockTwitterClient → pas besoin de credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from src.collector.config_loader import get_enabled_usernames
from src.collector.models import (
    AccountConfig,
    AccountsFileConfig,
    FiltersConfig,
    PollingConfig,
    RawTweet,
)
from src.collector.service import CollectorService
from src.collector.stocktwits_client import StockTwitsClient
from src.collector.twitter_client import MockTwitterClient
from src.collector.yfinance_news_client import YFinanceNewsClient
from src.core.database import Account, Tweet

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_accounts_config() -> AccountsFileConfig:
    return AccountsFileConfig(
        accounts=[
            AccountConfig(username="trader_a", priority="high", enabled=True),
            AccountConfig(username="trader_b", priority="medium", enabled=True),
            AccountConfig(username="trader_disabled", priority="low", enabled=False),
        ],
        polling=PollingConfig(interval_seconds=300, tweets_per_account=5),
        filters=FiltersConfig(
            exclude_retweets=True,
            exclude_replies=False,
            min_length=20,
            languages=["en"],
        ),
    )


@pytest.fixture
def mock_client() -> MockTwitterClient:
    return MockTwitterClient(seed=123)


def make_tweet(
    text: str = "BTC looking bullish, strong support here",
    username: str = "trader_a",
    is_retweet: bool = False,
    lang: str = "en",
    tickers: list[str] | None = None,
    sentiment: float | None = None,
    confidence: float | None = None,
) -> RawTweet:
    return RawTweet(
        tweet_id="123456789",
        username=username,
        text=text,
        tweeted_at=datetime.now(tz=timezone.utc),
        lang=lang,
        is_retweet=is_retweet,
        tickers=tickers or [],
        sentiment=sentiment,
        confidence=confidence,
    )


# ── Tests config_loader ────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_get_enabled_usernames_sorts_by_priority(self, sample_accounts_config):
        usernames = get_enabled_usernames(sample_accounts_config)
        assert usernames == ["trader_a", "trader_b"]   # disabled exclu
        assert "trader_disabled" not in usernames

    def test_get_enabled_usernames_excludes_disabled(self, sample_accounts_config):
        usernames = get_enabled_usernames(sample_accounts_config)
        assert len(usernames) == 2


# ── Tests MockTwitterClient ────────────────────────────────────────────────────

class TestMockTwitterClient:
    @pytest.mark.asyncio
    async def test_get_recent_tweets_returns_tweets(self, mock_client):
        tweets = await mock_client.get_recent_tweets("trader_a", max_results=5)
        assert isinstance(tweets, list)
        assert len(tweets) >= 1
        assert all(isinstance(t, RawTweet) for t in tweets)

    @pytest.mark.asyncio
    async def test_tweets_have_correct_username(self, mock_client):
        tweets = await mock_client.get_recent_tweets("my_trader", max_results=3)
        assert all(t.username == "my_trader" for t in tweets)

    @pytest.mark.asyncio
    async def test_resolve_user_id_returns_string(self, mock_client):
        uid = await mock_client.resolve_user_id("some_user")
        assert isinstance(uid, str)
        assert len(uid) > 0


# ── Tests filtres ─────────────────────────────────────────────────────────────

class TestCollectorFilters:
    def _make_service(self, config: AccountsFileConfig) -> CollectorService:
        return CollectorService(client=MockTwitterClient(), config=config)

    def test_filters_retweets(self, sample_accounts_config):
        svc = self._make_service(sample_accounts_config)
        rt = make_tweet(is_retweet=True)
        assert svc._should_keep(rt) is False

    def test_keeps_normal_tweet(self, sample_accounts_config):
        svc = self._make_service(sample_accounts_config)
        t = make_tweet()
        assert svc._should_keep(t) is True

    def test_filters_short_tweets(self, sample_accounts_config):
        svc = self._make_service(sample_accounts_config)
        short = make_tweet(text="gm")
        assert svc._should_keep(short) is False

    def test_filters_wrong_language(self, sample_accounts_config):
        svc = self._make_service(sample_accounts_config)
        fr_tweet = make_tweet(lang="fr")  # config: languages=["en"] seulement
        assert svc._should_keep(fr_tweet) is False

    def test_keeps_correct_language(self, sample_accounts_config):
        svc = self._make_service(sample_accounts_config)
        en_tweet = make_tweet(lang="en")
        assert svc._should_keep(en_tweet) is True


# ── Tests StockTwitsClient ──────────────────────────────────────────────────────

def _stocktwits_payload(messages: list[dict]) -> dict:
    return {"messages": messages}


def _stocktwits_message(
    msg_id: int = 1,
    body: str = "Loading up here, looks strong",
    author: str = "trader_x",
    symbols: list[str] | None = None,
    sentiment_label: str | None = "Bullish",
    created_at: str = "2026-08-06T10:00:00Z",
    likes: int = 3,
) -> dict:
    return {
        "id": msg_id,
        "body": body,
        "created_at": created_at,
        "user": {"username": author},
        "symbols": [{"symbol": s} for s in (symbols or ["BTC.X"])],
        "entities": {"sentiment": {"basic": sentiment_label} if sentiment_label else {}},
        "likes": {"total": likes},
    }


class TestStockTwitsClient:
    def _mock_response(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = payload
        return resp

    @pytest.mark.asyncio
    async def test_parses_messages_into_raw_tweets(self):
        payload = _stocktwits_payload([_stocktwits_message()])
        client = StockTwitsClient()

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(payload))):
            tweets = await client.get_recent_tweets("BTC.X", max_results=10)

        assert len(tweets) == 1
        tweet = tweets[0]
        assert tweet.tweet_id == "st_1"
        assert "trader_x" in tweet.text
        assert tweet.tickers == ["BTC.X"]

    @pytest.mark.asyncio
    async def test_maps_bullish_sentiment(self):
        payload = _stocktwits_payload([_stocktwits_message(sentiment_label="Bullish")])
        client = StockTwitsClient()

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(payload))):
            tweets = await client.get_recent_tweets("BTC.X")

        assert tweets[0].sentiment == pytest.approx(0.6)
        assert tweets[0].confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_maps_bearish_sentiment(self):
        payload = _stocktwits_payload([_stocktwits_message(sentiment_label="Bearish")])
        client = StockTwitsClient()

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(payload))):
            tweets = await client.get_recent_tweets("BTC.X")

        assert tweets[0].sentiment == pytest.approx(-0.6)

    @pytest.mark.asyncio
    async def test_no_declared_sentiment_leaves_none(self):
        payload = _stocktwits_payload([_stocktwits_message(sentiment_label=None)])
        client = StockTwitsClient()

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(payload))):
            tweets = await client.get_recent_tweets("BTC.X")

        assert tweets[0].sentiment is None
        assert tweets[0].confidence is None

    @pytest.mark.asyncio
    async def test_multiple_tickers_on_one_message(self):
        payload = _stocktwits_payload([_stocktwits_message(symbols=["BTC.X", "ETH.X"])])
        client = StockTwitsClient()

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(payload))):
            tweets = await client.get_recent_tweets("BTC.X")

        assert tweets[0].tickers == ["BTC.X", "ETH.X"]

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_list(self):
        client = StockTwitsClient()
        with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=Exception("network error"))):
            tweets = await client.get_recent_tweets("BTC.X")
        assert tweets == []

    @pytest.mark.asyncio
    async def test_resolve_user_id_returns_none(self):
        client = StockTwitsClient()
        assert await client.resolve_user_id("BTC.X") is None


# ── Tests persistance des hints (tickers/sentiment/confidence) ─────────────────

class TestSaveTweetsPersistsHints:
    @pytest.mark.asyncio
    async def test_pre_filled_hints_are_persisted(self, test_db):
        config = AccountsFileConfig(
            accounts=[AccountConfig(username="BTC.X", priority="high", enabled=True)],
        )
        svc = CollectorService(client=StockTwitsClient(), config=config)
        await svc.sync_accounts_to_db()

        raw = make_tweet(
            username="BTC.X", tickers=["BTC.X"], sentiment=0.6, confidence=0.6,
        )

        async with test_db() as session:
            account = (await session.execute(select(Account).where(Account.username == "BTC.X"))).scalar_one()
            await svc._save_tweets(session, [raw], account.id)
            await session.commit()

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet).where(Tweet.tweet_id == raw.tweet_id))).scalar_one()

        assert json.loads(tweet.tickers) == ["BTC.X"]
        assert tweet.sentiment == pytest.approx(0.6)
        assert tweet.confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_no_hints_leaves_fields_none_for_nlp(self, test_db):
        config = AccountsFileConfig(
            accounts=[AccountConfig(username="trader_a", priority="high", enabled=True)],
        )
        svc = CollectorService(client=MockTwitterClient(), config=config)
        await svc.sync_accounts_to_db()

        raw = make_tweet(username="trader_a")  # pas de tickers/sentiment/confidence

        async with test_db() as session:
            account = (await session.execute(select(Account).where(Account.username == "trader_a"))).scalar_one()
            await svc._save_tweets(session, [raw], account.id)
            await session.commit()

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet).where(Tweet.tweet_id == raw.tweet_id))).scalar_one()

        assert tweet.tickers is None
        assert tweet.sentiment is None
        assert tweet.confidence is None


# ── Tests YFinanceNewsClient ────────────────────────────────────────────────────

def _fake_article(
    uid: str = "abc123",
    title: str = "SK Hynix beats earnings estimates",
    publish_time: int | None = None,
) -> dict:
    return {
        "uuid": uid,
        "title": title,
        "providerPublishTime": publish_time if publish_time is not None else int(datetime.now(tz=timezone.utc).timestamp()),
    }


class TestYFinanceNewsClient:
    @pytest.mark.asyncio
    async def test_parses_articles_into_raw_tweets(self):
        client = YFinanceNewsClient()
        mock_ticker = MagicMock()
        mock_ticker.news = [_fake_article()]

        with patch("src.collector.yfinance_news_client.yf.Ticker", return_value=mock_ticker):
            tweets = await client.get_recent_tweets("000660.KS", max_results=20)

        assert len(tweets) == 1
        tweet = tweets[0]
        assert tweet.tweet_id == "yn_abc123"
        assert tweet.tickers == ["000660.KS"]
        assert tweet.sentiment is None
        assert tweet.confidence is None
        assert "earnings" in tweet.text

    @pytest.mark.asyncio
    async def test_skips_articles_without_title(self):
        client = YFinanceNewsClient()
        mock_ticker = MagicMock()
        mock_ticker.news = [{"uuid": "no-title"}]

        with patch("src.collector.yfinance_news_client.yf.Ticker", return_value=mock_ticker):
            tweets = await client.get_recent_tweets("AAPL")

        assert tweets == []

    @pytest.mark.asyncio
    async def test_respects_since_cutoff(self):
        client = YFinanceNewsClient()
        old_time = int((datetime.now(tz=timezone.utc)).timestamp()) - 3600 * 24 * 3
        mock_ticker = MagicMock()
        mock_ticker.news = [_fake_article(uid="old", publish_time=old_time)]

        with patch("src.collector.yfinance_news_client.yf.Ticker", return_value=mock_ticker):
            tweets = await client.get_recent_tweets("AAPL", since=datetime.now(tz=timezone.utc) - timedelta(hours=1))

        assert tweets == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        client = YFinanceNewsClient()
        with patch("src.collector.yfinance_news_client.yf.Ticker", side_effect=Exception("network error")):
            tweets = await client.get_recent_tweets("AAPL")
        assert tweets == []

    @pytest.mark.asyncio
    async def test_resolve_user_id_returns_none(self):
        client = YFinanceNewsClient()
        assert await client.resolve_user_id("AAPL") is None
