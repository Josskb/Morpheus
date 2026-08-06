"""
alerts/recap_service.py
─────────────────────────
Service de récap Telegram périodique : remplace les alertes individuelles
par 3 messages groupés à intervalles différents (moins bruyant, éprouvé en
prod sur la VM depuis plusieurs jours) :
  - récap "signaux forts" (par compte + par ticker), toutes les
    ALERT_RECAP_INTERVAL_HOURS
  - digest des signaux sous le seuil, toutes les DIGEST_INTERVAL_MINUTES
  - récap quotidien des titres les plus mentionnés, une fois/jour à
    DAILY_RECAP_HOUR_UTC

Indépendant d'AlertService (alertes individuelles à cooldown, conservées
telles quelles) — les deux peuvent tourner en parallèle.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import func, select

from ..core.database import Account, AsyncSessionLocal, Tweet
from ..core.settings import get_settings
from .bot import TelegramBot


class RecapService:
    """Boucle qui déclenche les 3 récaps Telegram selon leur propre intervalle."""

    def __init__(
        self,
        bot: TelegramBot | None = None,
        check_interval_seconds: int = 60,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._check_interval = check_interval_seconds
        self._running = False

        now = datetime.now(tz=timezone.utc)
        self._last_alert_recap_at = now
        self._last_digest_at = now
        self._last_daily_recap_date: date | None = None

    async def _send_alert_recap_if_due(self) -> None:
        """
        Toutes les ALERT_RECAP_INTERVAL_HOURS, envoie deux messages :
        signaux forts groupés par compte, puis agrégés par ticker.
        """
        now = datetime.now(tz=timezone.utc)
        interval = timedelta(hours=self._settings.alert_recap_interval_hours)
        if now - self._last_alert_recap_at < interval:
            return

        since = self._last_alert_recap_at
        threshold = self._settings.min_confidence_score

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Account.username, Tweet.text, Tweet.sentiment, Account.reliability_score)
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,  # noqa: E712
                    func.abs(Tweet.sentiment) >= threshold,
                    Tweet.created_at >= since,
                )
                .order_by(Account.username, func.abs(Tweet.sentiment).desc())
            )
            rows_by_account = result.all()

            # Pour les tweets StockTwits, Account.username contient le ticker
            # lui-même (pas un pseudo) — grouper par compte revient à grouper
            # par titre pour cette source.
            result = await session.execute(
                select(
                    Account.username,
                    func.count().label("count"),
                    func.avg(Tweet.sentiment).label("avg_sentiment"),
                )
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,  # noqa: E712
                    func.abs(Tweet.sentiment) >= threshold,
                    Tweet.created_at >= since,
                    Tweet.tweet_id.like("st_%"),
                )
                .group_by(Account.username)
                .order_by(func.count().desc())
            )
            rows_by_ticker = result.all()

        hours = self._settings.alert_recap_interval_hours
        await self._bot.send_recap_by_account(rows_by_account, hours)
        await self._bot.send_recap_by_ticker(rows_by_ticker, hours)
        self._last_alert_recap_at = now

    async def _send_digest_if_due(self) -> None:
        """Envoie un récap périodique des tweets sous le seuil d'alerte."""
        now = datetime.now(tz=timezone.utc)
        interval = timedelta(minutes=self._settings.digest_interval_minutes)
        if now - self._last_digest_at < interval:
            return

        since = self._last_digest_at
        threshold = self._settings.min_confidence_score

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Account.username, Tweet.text, Tweet.sentiment)
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,  # noqa: E712
                    func.abs(Tweet.sentiment) < threshold,
                    Tweet.created_at >= since,
                )
                .order_by(Account.username, Tweet.created_at.asc())
            )
            rows = result.all()

        await self._bot.send_digest(rows)
        self._last_digest_at = now

    async def _send_daily_recap_if_due(self) -> None:
        """
        Une fois par jour, à partir de DAILY_RECAP_HOUR_UTC, envoie le récap
        des titres les plus mentionnés sur les dernières 24h.
        """
        now = datetime.now(tz=timezone.utc)
        if now.hour < self._settings.daily_recap_hour_utc:
            return
        if self._last_daily_recap_date == now.date():
            return

        since = now - timedelta(days=1)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    Account.username,
                    func.count().label("mentions"),
                    func.avg(Tweet.sentiment).label("avg_sentiment"),
                )
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,  # noqa: E712
                    Tweet.sentiment.isnot(None),
                    Tweet.tickers.isnot(None),
                    Tweet.created_at >= since,
                )
                .group_by(Account.username)
                .order_by(func.count().desc())
                .limit(10)
            )
            rows = result.all()

        await self._bot.send_daily_recap(rows)
        self._last_daily_recap_date = now.date()

    async def run(self) -> None:
        """Boucle principale : vérifie les 3 récaps à chaque tick."""
        self._running = True
        await self._bot.start()
        logger.info(
            "RecapService démarré (alertes={}h, digest={}min, quotidien={}h UTC).",
            self._settings.alert_recap_interval_hours,
            self._settings.digest_interval_minutes,
            self._settings.daily_recap_hour_utc,
        )

        while self._running:
            try:
                await self._send_alert_recap_if_due()
                await self._send_digest_if_due()
                await self._send_daily_recap_if_due()
            except Exception as e:
                logger.exception("Erreur dans la boucle de récap : {}", e)
            await asyncio.sleep(self._check_interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("RecapService arrêté.")
