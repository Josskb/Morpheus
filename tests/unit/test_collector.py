"""
tests/unit/test_collector.py
─────────────────────────────
Tests unitaires du module collector.
Utilise MockTwitterClient → pas besoin de credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.collector.config_loader import get_enabled_usernames
from src.collector.models import (
    AccountConfig,
    AccountsFileConfig,
    FiltersConfig,
    PollingConfig,
    RawTweet,
)
from src.collector.service import CollectorService
from src.collector.twitter_client import MockTwitterClient

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
) -> RawTweet:
    return RawTweet(
        tweet_id="123456789",
        username=username,
        text=text,
        tweeted_at=datetime.now(tz=timezone.utc),
        lang=lang,
        is_retweet=is_retweet,
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
