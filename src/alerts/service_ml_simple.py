"""
Simplified ML Alert Service - Production Compatible
Version compatible avec le schema existant (sans alerted_at)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from src.core.database import Account, AsyncSessionLocal, Tweet
from src.core.settings import get_settings
from .telegram_bot import TelegramBot
from .formatter_ml import format_alert_with_ml, get_ml_alert_score

_ACTIONABLE_CALLS = {"long", "short"}


class AlertServiceMLSimple:
    """
    Simplified AlertService with ML scoring.
    Compatible avec schema existant (pas de colonne alerted_at).
    """

    def __init__(
        self,
        bot: TelegramBot | None = None,
        interval_seconds: int = 30,
        min_ml_score: float = 0.65,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._interval = interval_seconds
        self._running = False
        self._min_ml_score = min_ml_score
        self._alerted_tweets = set()  # Track sent tweets in memory

    async def check_and_send(self) -> int:
        """
        Check eligible tweets and send alerts with ML scoring.
        """
        threshold = self._settings.min_confidence_score
        sent = 0

        async with AsyncSessionLocal() as session:
            # Get tweets that haven't been alerted
            result = await session.execute(
                select(Tweet, Account)
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,  # noqa: E712
                    Tweet.confidence >= threshold,
                )
                .order_by(
                    Tweet.confidence.desc(),
                )
                .limit(50)
            )
            candidates = result.all()

            for tweet, account in candidates:
                # Skip if already sent
                if tweet.id in self._alerted_tweets:
                    continue

                # ML Scoring
                ml_score = get_ml_alert_score(tweet, account)

                if ml_score < self._min_ml_score:
                    logger.debug(
                        "ML score too low for @{}: {:.2f} < {:.2f}",
                        account.username,
                        ml_score,
                        self._min_ml_score,
                    )
                    continue

                # Format with ML
                message = format_alert_with_ml(tweet, account)
                ok = await self._bot.send(message)

                # Mark as sent
                self._alerted_tweets.add(tweet.id)
                sent += 1

                if ok:
                    logger.info(
                        "ML Alert sent: @{} | Score: {:.2f} | Conf: {:.2f}",
                        account.username,
                        ml_score,
                        tweet.confidence,
                    )
                else:
                    logger.info(
                        "ML Alert logged: @{} | Score: {:.2f}",
                        account.username,
                        ml_score,
                    )

        return sent

    async def run(self) -> None:
        """Main alert loop."""
        self._running = True
        await self._bot.start()
        logger.info(
            "AlertServiceMLSimple started (min_ml_score={:.2f})",
            self._min_ml_score,
        )

        while self._running:
            try:
                count = await self.check_and_send()
                if count:
                    logger.info("{} alert(s) sent via ML.", count)
            except Exception as e:
                logger.exception("Error in alert loop: {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("AlertServiceMLSimple stopped.")
