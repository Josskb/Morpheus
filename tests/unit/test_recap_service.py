"""
tests/unit/test_recap_service.py
───────────────────────────────────
Tests unitaires de RecapService : sélection des lignes (SQLAlchemy) et
logique de déclenchement temporel (_if_due). Le bot Telegram est mocké —
seul le contenu passé à ses méthodes de récap est vérifié.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from src.alerts.recap_service import RecapService
from src.core.database import Account, Tweet


async def _insert_tweet(
    session_factory,
    tweet_id: str,
    username: str,
    sentiment: float,
    tweeted_hours_ago: float = 0.0,
    created_hours_ago: float = 0.0,
    reliability_score: float | None = None,
) -> None:
    async with session_factory() as session:
        result = await session.execute(
            select(Account).where(Account.username == username)
        )
        account = result.scalar_one_or_none()
        if account is None:
            account = Account(
                username=username, markets="[]", tags="[]",
                priority="medium", enabled=True, reliability_score=reliability_score,
            )
            session.add(account)
            await session.flush()

        now = datetime.now(tz=timezone.utc)
        tweet = Tweet(
            tweet_id=tweet_id,
            account_id=account.id,
            text=f"signal for {username}",
            sentiment=sentiment,
            nlp_processed=True,
            tickers='["BTC"]',
            tweeted_at=now - timedelta(hours=tweeted_hours_ago),
        )
        session.add(tweet)
        await session.commit()

        # created_at a un server_default — on le force après coup si besoin
        # pour simuler un tweet plus ancien que le dernier récap.
        if created_hours_ago:
            tweet.created_at = now - timedelta(hours=created_hours_ago)
            session.add(tweet)
            await session.commit()


def _make_service(min_confidence: float = 0.65) -> RecapService:
    bot = AsyncMock()
    bot.send_recap_by_account = AsyncMock(return_value=True)
    bot.send_recap_by_ticker = AsyncMock(return_value=True)
    bot.send_digest = AsyncMock(return_value=True)
    bot.send_daily_recap = AsyncMock(return_value=True)
    svc = RecapService(bot=bot, check_interval_seconds=60)
    svc._settings.min_confidence_score = min_confidence
    return svc


class TestAlertRecap:
    @pytest.mark.asyncio
    async def test_strong_signal_included_weak_excluded(self, test_db):
        await _insert_tweet(test_db, "A1", "trader_strong", sentiment=0.9)
        await _insert_tweet(test_db, "A2", "trader_weak", sentiment=0.1)

        svc = _make_service(min_confidence=0.65)
        svc._last_alert_recap_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        await svc._send_alert_recap_if_due()

        rows_by_account = svc._bot.send_recap_by_account.call_args[0][0]
        usernames = {r[0] for r in rows_by_account}
        assert "trader_strong" in usernames
        assert "trader_weak" not in usernames

    @pytest.mark.asyncio
    async def test_ticker_recap_only_stocktwits_sourced(self, test_db):
        await _insert_tweet(test_db, "st_1", "BTC.X", sentiment=0.9)
        await _insert_tweet(test_db, "TW_1", "twitter_trader", sentiment=0.9)

        svc = _make_service(min_confidence=0.65)
        svc._last_alert_recap_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        await svc._send_alert_recap_if_due()

        rows_by_ticker = svc._bot.send_recap_by_ticker.call_args[0][0]
        usernames = {r[0] for r in rows_by_ticker}
        assert "BTC.X" in usernames
        assert "twitter_trader" not in usernames

    @pytest.mark.asyncio
    async def test_gating_skips_before_interval_elapsed(self, test_db):
        await _insert_tweet(test_db, "A1", "trader_strong", sentiment=0.9)

        svc = _make_service(min_confidence=0.65)
        svc._last_alert_recap_at = datetime.now(tz=timezone.utc)  # vient d'être envoyé
        await svc._send_alert_recap_if_due()

        svc._bot.send_recap_by_account.assert_not_called()

    @pytest.mark.asyncio
    async def test_excludes_signals_before_last_recap(self, test_db):
        await _insert_tweet(
            test_db, "A1", "old_signal", sentiment=0.9, created_hours_ago=5,
        )

        svc = _make_service(min_confidence=0.65)
        svc._last_alert_recap_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        await svc._send_alert_recap_if_due()

        rows_by_account = svc._bot.send_recap_by_account.call_args[0][0]
        usernames = {r[0] for r in rows_by_account}
        assert "old_signal" not in usernames


class TestDigest:
    @pytest.mark.asyncio
    async def test_weak_signal_included_strong_excluded(self, test_db):
        await _insert_tweet(test_db, "A1", "trader_strong", sentiment=0.9)
        await _insert_tweet(test_db, "A2", "trader_weak", sentiment=0.1)

        svc = _make_service(min_confidence=0.65)
        svc._last_digest_at = datetime.now(tz=timezone.utc) - timedelta(minutes=45)
        await svc._send_digest_if_due()

        rows = svc._bot.send_digest.call_args[0][0]
        usernames = {r[0] for r in rows}
        assert "trader_weak" in usernames
        assert "trader_strong" not in usernames

    @pytest.mark.asyncio
    async def test_gating_respects_interval(self, test_db):
        await _insert_tweet(test_db, "A1", "trader_weak", sentiment=0.1)

        svc = _make_service(min_confidence=0.65)
        svc._last_digest_at = datetime.now(tz=timezone.utc)
        await svc._send_digest_if_due()

        svc._bot.send_digest.assert_not_called()


class TestDailyRecap:
    @pytest.mark.asyncio
    async def test_limited_to_top_10_by_mentions(self, test_db):
        for i in range(12):
            for j in range(12 - i):  # plus de mentions pour les premiers usernames
                await _insert_tweet(test_db, f"D{i}-{j}", f"ticker_{i}", sentiment=0.3)

        svc = _make_service()
        svc._settings.daily_recap_hour_utc = 0  # toujours dû (heure UTC courante >= 0)
        svc._last_daily_recap_date = None
        await svc._send_daily_recap_if_due()

        rows = svc._bot.send_daily_recap.call_args[0][0]
        assert len(rows) <= 10

    @pytest.mark.asyncio
    async def test_skips_before_configured_hour(self, test_db):
        await _insert_tweet(test_db, "A1", "ticker_a", sentiment=0.3)

        svc = _make_service()
        svc._settings.daily_recap_hour_utc = 23  # improbable d'être déjà 23h UTC
        svc._last_daily_recap_date = None
        now_hour = datetime.now(tz=timezone.utc).hour
        if now_hour >= 23:
            pytest.skip("test non déterministe à cette heure précise")
        await svc._send_daily_recap_if_due()

        svc._bot.send_daily_recap.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_once_per_day(self, test_db):
        await _insert_tweet(test_db, "A1", "ticker_a", sentiment=0.3)

        svc = _make_service()
        svc._settings.daily_recap_hour_utc = 0
        svc._last_daily_recap_date = datetime.now(tz=timezone.utc).date()  # déjà envoyé aujourd'hui
        await svc._send_daily_recap_if_due()

        svc._bot.send_daily_recap.assert_not_called()
