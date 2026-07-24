"""
Hybrid ML+NLP Alert Service (Optimized)
Balanced approach combining NLP confidence with ML predictions
"""

from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import select

from src.core.database import Account, AsyncSessionLocal, Tweet
from src.core.settings import get_settings
from src.ml.hybrid_scorer import HybridScorer
from .telegram_bot import TelegramBot
from .formatter_ml import format_alert_with_ml


class AlertServiceMLHybrid:
    """
    Optimized alert service using hybrid NLP+ML scoring.

    Threshold: 0.50 (optimized based on score distribution analysis)
    Weighting: 70% NLP confidence + 30% ML predictions
    """

    def __init__(
        self,
        bot: TelegramBot | None = None,
        interval_seconds: int = 30,
        min_hybrid_score: float = 0.50,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._interval = interval_seconds
        self._running = False
        self._min_hybrid_score = min_hybrid_score
        self._alerted_tweets = set()
        self._scorer = HybridScorer(nlp_weight=0.70, ml_weight=0.30)

    async def check_and_send(self) -> dict:
        """
        Check eligible tweets and send alerts with hybrid scoring.

        Returns:
            {
                "sent": int,  # Alerts sent
                "evaluated": int,  # Tweets evaluated
                "filtered_by_score": int,  # Filtered by low hybrid score
            }
        """
        threshold = self._settings.min_confidence_score
        sent = 0
        filtered = 0
        evaluated = 0

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet, Account)
                .join(Account, Tweet.account_id == Account.id)
                .where(Tweet.nlp_processed == True)  # noqa: E712
                .order_by(Tweet.confidence.desc())
                .limit(100)
            )
            candidates = result.all()

            for tweet, account in candidates:
                evaluated += 1

                # Skip if already sent
                if tweet.id in self._alerted_tweets:
                    continue

                # Hybrid scoring
                tweet_data = {
                    "confidence": tweet.confidence or 0.0,
                    "sentiment": tweet.sentiment or 0.0,
                    "urgency_score": tweet.urgency_score or 0.0,
                    "like_count": tweet.like_count or 0,
                    "retweet_count": tweet.retweet_count or 0,
                    "reply_count": tweet.reply_count or 0,
                }

                account_data = {
                    "reliability_score": account.reliability_score or 0.5,
                    "win_rate": account.win_rate or 0.5,
                }

                score_result = self._scorer.score(tweet_data, account_data)
                hybrid_score = score_result["hybrid_score"]

                if hybrid_score < self._min_hybrid_score:
                    filtered += 1
                    continue

                # Format with ML info
                message = format_alert_with_ml(tweet, account)
                ok = await self._bot.send_message(message)

                # Mark as sent
                self._alerted_tweets.add(tweet.id)
                sent += 1

                if ok:
                    logger.info(
                        "ALERT SENT: @{} | Hybrid={:.3f} (NLP={:.2f}, ML={:.2f}) | Rec: {}",
                        account.username,
                        hybrid_score,
                        score_result["nlp_score"],
                        score_result["ml_score"],
                        score_result["recommendation"],
                    )
                else:
                    logger.info(
                        "Alert logged: @{} | Hybrid={:.3f}",
                        account.username,
                        hybrid_score,
                    )

        return {
            "sent": sent,
            "evaluated": evaluated,
            "filtered_by_score": filtered,
        }

    async def run(self) -> None:
        """Main alert loop."""
        self._running = True
        await self._bot.start()
        logger.info(
            "AlertServiceMLHybrid started (min_score={:.2f}, weights: 70% NLP + 30% ML)",
            self._min_hybrid_score,
        )

        while self._running:
            try:
                result = await self.check_and_send()
                if result["sent"]:
                    logger.info(
                        "{} alert(s) sent (evaluated: {}, filtered: {})",
                        result["sent"],
                        result["evaluated"],
                        result["filtered_by_score"],
                    )
            except Exception as e:
                logger.exception("Error in alert loop: {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("AlertServiceMLHybrid stopped.")
