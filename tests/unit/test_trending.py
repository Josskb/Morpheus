"""
tests/unit/test_trending.py
──────────────────────────────
Tests unitaires de TrendingService : détection de spike de mentions,
cooldown persisté, capture/suivi de prix (TrendingSnapshot), et le rapport
agrégé spike → mouvement de prix.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from src.collector.trending_service import TrendingService, _fetcher_ticker
from src.core.database import TrendingSnapshot


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


def _mock_fetcher(price: float | None = 100.0):
    fetcher = MagicMock()
    fetcher.get_price = AsyncMock(return_value=(price, "crypto"))
    return patch("src.collector.trending_service.get_market_fetcher", return_value=fetcher)


class TestFetcherTicker:
    def test_strips_x_suffix_for_crypto(self):
        assert _fetcher_ticker("BTC.X", "crypto") == "BTC"

    def test_keeps_stock_ticker_unchanged(self):
        assert _fetcher_ticker("SNDK", "stock") == "SNDK"


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
    async def test_sends_alert_for_notable_ticker(self, tmp_path, test_db):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}],
            [],
        ])

        with _mock_fetcher(price=42000.0):
            sent = await svc.check_once()

        assert sent == 1
        svc._bot.send_trending_alert.assert_called_once_with(
            ticker="BTC.X", name="Bitcoin", mentions=100, mentions_24h_ago=10, market="Crypto",
        )
        assert (tmp_path / "trending_seen.json").exists()

    @pytest.mark.asyncio
    async def test_creates_snapshot_with_price_at_detection(self, tmp_path, test_db):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}],
            [],
        ])

        with _mock_fetcher(price=42000.0):
            await svc.check_once()

        async with test_db() as session:
            snap = (await session.execute(select(TrendingSnapshot))).scalar_one()

        assert snap.ticker == "BTC.X"
        assert snap.market_type == "crypto"
        assert snap.price_at_detection == 42000.0
        assert snap.mentions_at_detection == 100
        assert snap.mentions_24h_ago_at_detection == 10

    @pytest.mark.asyncio
    async def test_no_notable_ticker_sends_nothing(self, tmp_path, test_db):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 15, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 1}],
            [],
        ])

        with _mock_fetcher():
            sent = await svc.check_once()

        assert sent == 0
        svc._bot.send_trending_alert.assert_not_called()
        assert not (tmp_path / "trending_seen.json").exists()

        async with test_db() as session:
            snaps = (await session.execute(select(TrendingSnapshot))).scalars().all()
        assert snaps == []

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate_snapshot(self, tmp_path, test_db):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0, cooldown_hours=6)
        entry = [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}]
        svc._client.get_trending = AsyncMock(side_effect=[entry, [], entry, []])

        with _mock_fetcher(price=42000.0):
            first = await svc.check_once()
            second = await svc.check_once()

        assert first == 1
        assert second == 0  # cooldown actif, pas de deuxième alerte ni de deuxième snapshot

        async with test_db() as session:
            snaps = (await session.execute(select(TrendingSnapshot))).scalars().all()
        assert len(snaps) == 1

    @pytest.mark.asyncio
    async def test_checks_both_crypto_and_stocks_filters(self, tmp_path, test_db):
        svc = _make_service(tmp_path, min_mentions=10, spike_multiplier=3.0)
        svc._client.get_trending = AsyncMock(side_effect=[
            [{"ticker": "BTC.X", "name": "Bitcoin", "mentions": 100, "mentions_24h_ago": 10, "rank": 1, "rank_24h_ago": 5}],
            [{"ticker": "AAPL", "name": "Apple", "mentions": 50, "mentions_24h_ago": 5, "rank": 1, "rank_24h_ago": 3}],
        ])

        with _mock_fetcher():
            sent = await svc.check_once()

        assert sent == 2
        calls = svc._client.get_trending.call_args_list
        assert calls[0].args[0] == "all-crypto"
        assert calls[1].args[0] == "all-stocks"


class TestUpdatePendingSnapshots:
    async def _insert_snapshot(self, session_factory, detected_hours_ago: float, **overrides) -> None:
        async with session_factory() as session:
            defaults = dict(
                ticker="BTC.X", market_type="crypto",
                mentions_at_detection=100, mentions_24h_ago_at_detection=10,
                detected_at=datetime.now(tz=timezone.utc) - timedelta(hours=detected_hours_ago),
                price_at_detection=100.0,
            )
            defaults.update(overrides)
            session.add(TrendingSnapshot(**defaults))
            await session.commit()

    @pytest.mark.asyncio
    async def test_fills_due_window(self, tmp_path, test_db):
        svc = _make_service(tmp_path)
        await self._insert_snapshot(test_db, detected_hours_ago=2)  # 1h due, 4h/24h/7d not yet

        with _mock_fetcher(price=110.0):
            await svc._update_pending_snapshots()

        async with test_db() as session:
            snap = (await session.execute(select(TrendingSnapshot))).scalar_one()

        assert snap.price_1h == 110.0
        assert snap.change_1h == pytest.approx(10.0)
        assert snap.change_4h is None  # pas encore due

    @pytest.mark.asyncio
    async def test_does_not_touch_too_recent_snapshot(self, tmp_path, test_db):
        svc = _make_service(tmp_path)
        await self._insert_snapshot(test_db, detected_hours_ago=0.1)  # 6 min, aucune fenêtre due

        with _mock_fetcher(price=999.0):
            await svc._update_pending_snapshots()

        async with test_db() as session:
            snap = (await session.execute(select(TrendingSnapshot))).scalar_one()
        assert snap.change_1h is None

    @pytest.mark.asyncio
    async def test_does_not_refetch_already_filled_window(self, tmp_path, test_db):
        svc = _make_service(tmp_path)
        await self._insert_snapshot(
            test_db, detected_hours_ago=2, price_1h=105.0, change_1h=5.0,
        )

        fetcher = MagicMock()
        fetcher.get_price = AsyncMock(return_value=(999.0, "crypto"))
        with patch("src.collector.trending_service.get_market_fetcher", return_value=fetcher):
            await svc._update_pending_snapshots()

        async with test_db() as session:
            snap = (await session.execute(select(TrendingSnapshot))).scalar_one()
        assert snap.change_1h == 5.0  # inchangé, pas re-fetché
        fetcher.get_price.assert_not_called()


class TestGenerateReport:
    async def _insert_resolved(self, session_factory, ticker: str, market_type: str, change_1h: float) -> None:
        async with session_factory() as session:
            session.add(TrendingSnapshot(
                ticker=ticker, market_type=market_type,
                mentions_at_detection=100, mentions_24h_ago_at_detection=10,
                detected_at=datetime.now(tz=timezone.utc) - timedelta(hours=2),
                price_at_detection=100.0, price_1h=100.0 * (1 + change_1h / 100),
                change_1h=change_1h,
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_report(self, tmp_path, test_db):
        svc = _make_service(tmp_path)
        report = await svc.generate_report()
        assert report == {}

    @pytest.mark.asyncio
    async def test_computes_win_rate_and_avg_change(self, tmp_path, test_db):
        svc = _make_service(tmp_path)
        await self._insert_resolved(test_db, "BTC.X", "crypto", change_1h=10.0)
        await self._insert_resolved(test_db, "ETH.X", "crypto", change_1h=-2.0)
        await self._insert_resolved(test_db, "AAPL", "stock", change_1h=5.0)

        report = await svc.generate_report()

        assert report["1h"]["n"] == 3
        assert report["1h"]["win_rate"] == pytest.approx(2 / 3)
        assert report["1h"]["avg_change"] == pytest.approx((10.0 - 2.0 + 5.0) / 3)
        assert report["1h"]["by_market"]["crypto"]["n"] == 2
        assert report["1h"]["by_market"]["stock"]["n"] == 1
        assert "4h" not in report  # aucune donnée résolue sur cette fenêtre
