"""
tests/unit/test_alerts.py
──────────────────────────
Tests unitaires du module alerts (formatter + service).
Pas de Telegram réel, pas de DB — tout est mocké.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.alerts.formatter import _conf_label, _price_fmt, format_alert
from src.core.database import Account, Tweet

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_account(username: str = "crypto_trader") -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        username=username,
        display_name="Crypto Trader",
    )


def make_tweet(
    tweet_id: str = "123456789",
    call_type: str = "long",
    tickers: list[str] | None = None,
    confidence: float = 0.82,
    sentiment: float = 0.6,
    target_price: float | None = 50_000.0,
    stop_loss: float | None = 38_000.0,
    urgency_score: float = 0.3,
    text: str = "Loading up $BTC here, very bullish! Strong support 🚀",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        tweet_id=tweet_id,
        account_id=1,
        call_type=call_type,
        tickers=json.dumps(tickers or ["BTC"]),
        confidence=confidence,
        sentiment=sentiment,
        target_price=target_price,
        stop_loss=stop_loss,
        urgency_score=urgency_score,
        text=text,
        nlp_processed=True,
        alerted_at=None,
        tweeted_at=datetime.now(tz=timezone.utc),
    )


# ── formatter ─────────────────────────────────────────────────────────────────

class TestFormatter:
    def test_format_long_tweet(self):
        tweet = make_tweet(call_type="long")
        account = make_account()
        msg = format_alert(tweet, account)

        assert "LONG" in msg
        assert "📈" in msg
        assert "BTC" in msg
        assert "@crypto_trader" in msg
        assert "0.82" in msg

    def test_format_short_tweet(self):
        tweet = make_tweet(call_type="short")
        account = make_account()
        msg = format_alert(tweet, account)

        assert "SHORT" in msg
        assert "📉" in msg

    def test_format_includes_target_price(self):
        tweet = make_tweet(target_price=50_000.0)
        msg = format_alert(tweet, make_account())
        assert "50,000" in msg or "50000" in msg
        assert "Cible" in msg

    def test_format_includes_stop_loss(self):
        tweet = make_tweet(stop_loss=38_000.0)
        msg = format_alert(tweet, make_account())
        assert "38,000" in msg or "38000" in msg
        assert "Stop" in msg

    def test_format_no_target_no_section(self):
        tweet = make_tweet(target_price=None, stop_loss=None)
        msg = format_alert(tweet, make_account())
        assert "Cible" not in msg
        assert "Stop" not in msg

    def test_format_tweet_url(self):
        tweet = make_tweet(tweet_id="999888777")
        account = make_account(username="test_user")
        msg = format_alert(tweet, account)
        assert "twitter.com/test_user/status/999888777" in msg

    def test_format_multiple_tickers(self):
        tweet = make_tweet(tickers=["BTC", "ETH", "SOL"])
        msg = format_alert(tweet, make_account())
        assert "BTC" in msg
        assert "ETH" in msg
        assert "SOL" in msg

    def test_format_long_text_truncated(self):
        long_text = "A" * 300
        tweet = make_tweet(text=long_text)
        msg = format_alert(tweet, make_account())
        assert "…" in msg

    def test_format_is_html(self):
        msg = format_alert(make_tweet(), make_account())
        assert "<b>" in msg
        assert "<i>" in msg
        assert "<a " in msg

    def test_conf_label_fort(self):
        assert _conf_label(0.90) == "FORT"
        assert _conf_label(0.80) == "FORT"

    def test_conf_label_moyen(self):
        assert _conf_label(0.72) == "MOYEN"

    def test_conf_label_faible(self):
        assert _conf_label(0.50) == "FAIBLE"

    def test_price_fmt_large(self):
        assert "$50,000" == _price_fmt(50_000)

    def test_price_fmt_small(self):
        result = _price_fmt(0.0045)
        assert "$0.0045" == result

    def test_price_fmt_mid(self):
        result = _price_fmt(2.50)
        assert "$2.50" == result


# ── AlertService ──────────────────────────────────────────────────────────────

class TestAlertService:
    """Tests du service d'alertes avec DB mockée."""

    def _make_service(self, mock_bot=None):
        from src.alerts.service import AlertService
        svc = AlertService.__new__(AlertService)
        svc._settings = MagicMock()
        svc._settings.min_confidence_score = 0.65
        svc._settings.alert_cooldown_minutes = 5
        svc._bot = mock_bot or AsyncMock()
        svc._bot.send = AsyncMock(return_value=True)
        svc._interval = 30
        svc._running = False
        return svc

    @pytest.mark.asyncio
    async def test_check_and_send_alerts_eligible_tweet(self, test_db):
        """Un tweet éligible déclenche une alerte."""
        from src.alerts.service import AlertService

        # Insère un compte + tweet éligible
        async with test_db() as session:
            account = Account(
                username="alert_trader", markets='[]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id="ALERT001",
                account_id=account.id,
                text="$BTC loading up here, bullish breakout confirmed! 🚀",
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                tickers='["BTC"]',
                call_type="long",
                confidence=0.85,
                sentiment=0.7,
                urgency_score=0.3,
                nlp_processed=True,
                alerted_at=None,
            )
            session.add(tweet)
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=True)
        svc = AlertService(bot=mock_bot)

        count = await svc.check_and_send()

        assert count == 1
        mock_bot.send.assert_called_once()

        # Vérifie que alerted_at est maintenant rempli
        async with test_db() as session:
            from sqlalchemy import select
            result = await session.execute(select(Tweet).where(Tweet.tweet_id == "ALERT001"))
            t = result.scalar_one()
        assert t.alerted_at is not None

    @pytest.mark.asyncio
    async def test_check_and_send_skips_low_confidence(self, test_db):
        """Un tweet sous le seuil de confiance ne déclenche pas d'alerte."""
        from src.alerts.service import AlertService

        async with test_db() as session:
            account = Account(
                username="low_conf_trader", markets='[]', tags='[]',
                priority="medium", enabled=True,
            )
            session.add(account)
            await session.flush()

            session.add(Tweet(
                tweet_id="LOWCONF001",
                account_id=account.id,
                text="$ETH might be bullish, not sure though, watching",
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                tickers='["ETH"]',
                call_type="long",
                confidence=0.30,   # sous le seuil 0.65
                nlp_processed=True,
                alerted_at=None,
            ))
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=True)
        svc = AlertService(bot=mock_bot)
        count = await svc.check_and_send()

        assert count == 0
        mock_bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_send_skips_info_tweets(self, test_db):
        """Les tweets 'info' ou 'unclear' ne déclenchent pas d'alerte."""
        from src.alerts.service import AlertService

        async with test_db() as session:
            account = Account(
                username="info_trader", markets='[]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            for call_type in ("info", "unclear", "hold"):
                session.add(Tweet(
                    tweet_id=f"INFO_{call_type}",
                    account_id=account.id,
                    text=f"$BTC {call_type} signal here watching carefully",
                    lang="en",
                    tweeted_at=datetime.now(tz=timezone.utc),
                    tickers='["BTC"]',
                    call_type=call_type,
                    confidence=0.90,
                    nlp_processed=True,
                    alerted_at=None,
                ))
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=True)
        svc = AlertService(bot=mock_bot)
        count = await svc.check_and_send()

        assert count == 0

    @pytest.mark.asyncio
    async def test_check_and_send_skips_already_alerted(self, test_db):
        """Un tweet avec alerted_at != None n'est pas renvoyé."""
        from src.alerts.service import AlertService

        async with test_db() as session:
            account = Account(
                username="already_alerted", markets='[]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            session.add(Tweet(
                tweet_id="ALERTED001",
                account_id=account.id,
                text="$BTC very bullish, loading up here strong support 🚀",
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                tickers='["BTC"]',
                call_type="long",
                confidence=0.88,
                nlp_processed=True,
                alerted_at=datetime.now(tz=timezone.utc),  # déjà alerté
            ))
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=True)
        svc = AlertService(bot=mock_bot)
        count = await svc.check_and_send()

        assert count == 0
        mock_bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_second_alert_same_account(self, test_db):
        """Deux tweets du même compte dans la fenêtre cooldown → une seule alerte."""
        from src.alerts.service import AlertService

        async with test_db() as session:
            account = Account(
                username="cooldown_trader", markets='[]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            now = datetime.now(tz=timezone.utc)
            for i, tweet_id in enumerate(["CD001", "CD002"]):
                session.add(Tweet(
                    tweet_id=tweet_id,
                    account_id=account.id,
                    text=f"$BTC very bullish loading up tweet number {i} 🚀",
                    lang="en",
                    tweeted_at=now - timedelta(minutes=i),
                    tickers='["BTC"]',
                    call_type="long",
                    confidence=0.85,
                    urgency_score=float(i),
                    nlp_processed=True,
                    alerted_at=None,
                ))
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=True)
        svc = AlertService(bot=mock_bot)
        count = await svc.check_and_send()

        # Cooldown de 5 min → seulement 1 alerte pour les 2 tweets du même compte
        assert count == 1
        mock_bot.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_still_marks_alerted(self, test_db):
        """Même si Telegram échoue, alerted_at est marqué pour éviter le spam."""
        from src.alerts.service import AlertService

        async with test_db() as session:
            account = Account(
                username="fail_trader", markets='[]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            session.add(Tweet(
                tweet_id="FAIL001",
                account_id=account.id,
                text="$BTC loading up bullish breakout here 🚀",
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                tickers='["BTC"]',
                call_type="long",
                confidence=0.85,
                nlp_processed=True,
                alerted_at=None,
            ))
            await session.commit()

        mock_bot = AsyncMock()
        mock_bot.send = AsyncMock(return_value=False)  # Telegram indisponible
        svc = AlertService(bot=mock_bot)
        count = await svc.check_and_send()

        assert count == 1  # compté comme traité

        async with test_db() as session:
            from sqlalchemy import select
            result = await session.execute(select(Tweet).where(Tweet.tweet_id == "FAIL001"))
            t = result.scalar_one()
        assert t.alerted_at is not None  # marqué pour ne pas respammer


# ── TelegramBot — récap (portés depuis la VM) ──────────────────────────────────

class TestTradingViewUrl:
    def test_crypto_suffix(self):
        from src.alerts.bot import _tradingview_url
        assert _tradingview_url("BTC.X") == "https://www.tradingview.com/symbols/BTCUSD/"

    def test_plain_us_ticker(self):
        from src.alerts.bot import _tradingview_url
        assert _tradingview_url("AAPL") == "https://www.tradingview.com/symbols/AAPL/"

    def test_international_ticker_returns_none(self):
        from src.alerts.bot import _tradingview_url
        assert _tradingview_url("000660.KS") is None
        assert _tradingview_url("ALKAL.PA") is None


class TestTelegramBotShortsellerAlert:
    def _make_bot(self):
        from src.alerts.bot import TelegramBot
        bot = TelegramBot.__new__(TelegramBot)
        bot._configured = False
        bot._bot = None
        bot.send = AsyncMock(return_value=True)
        return bot

    @pytest.mark.asyncio
    async def test_formats_shortseller_alert(self):
        bot = self._make_bot()
        await bot.send_shortseller_alert(
            firm="Grizzly Research", emoji="🐻",
            title="ACME Corp <fraud>", url="https://grizzlyreports.com/reports/acme",
        )
        text = bot.send.call_args[0][0]
        assert "NOUVEAU RAPPORT SHORT-SELLER" in text
        assert "Grizzly Research" in text
        assert "&lt;fraud&gt;" in text  # échappé
        assert "https://grizzlyreports.com/reports/acme" in text


class TestTelegramBotRecap:
    def _make_bot(self):
        from src.alerts.bot import TelegramBot
        bot = TelegramBot.__new__(TelegramBot)
        bot._configured = False
        bot._bot = None
        bot.send = AsyncMock(return_value=True)
        return bot

    @pytest.mark.asyncio
    async def test_send_digest_empty_rows_no_send(self):
        bot = self._make_bot()
        result = await bot.send_digest([])
        assert result is False
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_digest_groups_by_username(self):
        bot = self._make_bot()
        rows = [
            ("trader_a", "BTC looking good here", 0.1),
            ("trader_a", "still watching BTC", 0.02),
            ("trader_b", "not sure about ETH", -0.03),
        ]
        await bot.send_digest(rows)
        text = bot.send.call_args[0][0]
        assert "@trader_a" in text
        assert "@trader_b" in text
        assert text.count("@trader_a") == 1  # groupé, pas répété

    @pytest.mark.asyncio
    async def test_send_digest_truncates_and_escapes(self):
        bot = self._make_bot()
        long_text = "x" * 300
        await bot.send_digest([("trader_a", f"<script>{long_text}", 0.0)])
        text = bot.send.call_args[0][0]
        assert "&lt;script&gt;" in text
        assert "…" in text

    @pytest.mark.asyncio
    async def test_send_recap_by_account_empty_no_send(self):
        bot = self._make_bot()
        assert await bot.send_recap_by_account([], 2) is False
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_recap_by_account_star_for_reliable(self):
        bot = self._make_bot()
        rows = [
            ("reliable_trader", "BTC breakout", 0.8, 0.75),
            ("shaky_trader", "ETH breakdown", -0.7, 0.3),
        ]
        await bot.send_recap_by_account(rows, 2)
        text = bot.send.call_args[0][0]
        assert "@reliable_trader</b> ⭐" in text
        assert "@shaky_trader</b> ⭐" not in text
        assert "2h" in text

    @pytest.mark.asyncio
    async def test_send_recap_by_ticker_empty_no_send(self):
        bot = self._make_bot()
        assert await bot.send_recap_by_ticker([], 2) is False
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_recap_by_ticker_splits_crypto_and_stocks(self):
        bot = self._make_bot()
        rows = [("BTC.X", 5, 0.6), ("AAPL", 3, -0.2)]
        await bot.send_recap_by_ticker(rows, 2)
        text = bot.send.call_args[0][0]
        assert "Bourse" in text
        assert "Crypto" in text
        assert "AAPL" in text
        assert "BTC.X" in text

    @pytest.mark.asyncio
    async def test_send_recap_by_ticker_links_for_tradeable_tickers(self):
        bot = self._make_bot()
        rows = [("BTC.X", 1, 0.5)]
        await bot.send_recap_by_ticker(rows, 2)
        text = bot.send.call_args[0][0]
        assert 'href="https://www.tradingview.com/symbols/BTCUSD/"' in text

    @pytest.mark.asyncio
    async def test_send_recap_by_ticker_no_link_for_international(self):
        bot = self._make_bot()
        rows = [("000660.KS", 1, 0.5)]
        await bot.send_recap_by_ticker(rows, 2)
        text = bot.send.call_args[0][0]
        assert "<a href" not in text
        assert "000660.KS" in text

    @pytest.mark.asyncio
    async def test_send_daily_recap_empty_no_send(self):
        bot = self._make_bot()
        assert await bot.send_daily_recap([]) is False
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_daily_recap_formats_mentions(self):
        bot = self._make_bot()
        rows = [("BTC.X", 12, 0.4), ("000660.KS", 3, -0.1)]
        await bot.send_daily_recap(rows)
        text = bot.send.call_args[0][0]
        assert "12 mentions" in text
        assert "3 mentions" in text
        assert 'href="https://www.tradingview.com/symbols/BTCUSD/"' in text
        assert "000660.KS</b> — 3 mentions" in text  # pas de lien, pas de crash
