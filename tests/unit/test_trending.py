"""
tests/unit/test_trending.py
──────────────────────────────
Tests unitaires de TrendingService : détection de spike de mentions,
cooldown persisté, et la boucle de vérification (client/bot mockés).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.collector.trending_service import TrendingService


def _make_service(tmp_path, min_mentions: int = 10, spike_multiplier: float = 3.0, cooldown_hours: int = 6) -> TrendingService:
    mock_client = AsyncMock()
    mock_bot = AsyncMock()
    svc = TrendingService(
        client=mock_client, bot=mock_bot,
        state_file=tmp_path / "trending_seen.json",
    )
    svc._settings.trending_min_mentions = min_mentions
    svc._settings.trending_spike_multiplier = spike_multiplier
    svc._settings.trending_alert_cooldown_hours = cooldown_hours
    return svc


class TestIsNotable:
    def test_below_min_mentions_not_notable(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10)
        assert svc._is_notable(mentions=9, mentions_24h_ago=1) is False

    def test_spike_above_multiplier_is_notable(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        assert svc._is_notable(mentions=30, mentions_24h_ago=10) is True

    def test_below_multiplier_not_notable(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        assert svc._is_notable(mentions=20, mentions_24h_ago=10) is False

    def test_new_ticker_zero_baseline_is_notable(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10)
        assert svc._is_notable(mentions=15, mentions_24h_ago=0) is True

    def test_new_ticker_below_min_mentions_not_notable(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10)
        assert svc._is_notable(mentions=5, mentions_24h_ago=0) is False


class TestCooldown:
    def test_no_prior_alert_cooldown_elapsed(self, tmp_path):
        svc = _make_service(tmp_path)
        assert svc._cooldown_elapsed({}, "BTC.X") is True

    def test_recent_alert_cooldown_not_elapsed(self, tmp_path):
        svc = _make_service(tmp_path, cooldown_hours=6)
        state = {"BTC.X": datetime.now(tz=timezone.utc).isoformat()}
        assert svc._cooldown_elapsed(state, "BTC.X") is False

    def test_old_alert_cooldown_elapsed(self, tmp_path):
        svc = _make_service(tmp_path, cooldown_hours=6)
        old = datetime.now(tz=timezone.utc) - timedelta(hours=7)
        state = {"BTC.X": old.isoformat()}
        assert svc._cooldown_elapsed(state, "BTC.X") is True


class TestCheckOnce:
    @pytest.mark.asyncio
    async def test_sends_alert_for_notable_ticker(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}],
            [],
        ])

        sent = await svc.check_once()

        assert sent == 1
        svc._bot.send_trending_alert.assert_called_once_with(
            ticker="BTC.X", name="Bitcoin", mentions=100, mentions_24h_ago=10, market="Crypto",
        )
        assert (tmp_path / "trending_seen.json").exists()

    @pytest.mark.asyncio
    async def test_no_notable_ticker_sends_nothing(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 15, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 1}],
            [],
        ])

        sent = await svc.check_once()

        assert sent == 0
        svc._bot.send_trending_alert.assert_not_called()
        assert not (tmp_path / "trending_seen.json").exists()

    @pytest.mark.asyncio
    async def test_respects_cooldown_across_calls(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0, cooldown_hours=6)
        entry = [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}]
        svc._client.get_trending = AsyncMock(side_effect=[entry, [], entry, []])

        first = await svc.check_once()
        second = await svc.check_once()

        assert first == 1
        assert second == 0  # cooldown actif, pas de deuxième alerte

    @pytest.mark.asyncio
    async def test_checks_both_crypto_and_stocks_filters(self, tmp_path):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}],
            [{"ticker": "AAPL", "name": "Apple", "mentions": 50, "mentions_24h_ago": 5, "rank": 1, "rank_24h_ago": 3}],
        ])

        sent = await svc.check_once()

        assert sent == 2
        calls = svc._client.get_trending.call_args_list
        assert calls[0].args[0] == "all-crypto"
        assert calls[1].args[0] == "all-stocks"
