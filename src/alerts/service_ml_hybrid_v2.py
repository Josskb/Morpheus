"""
Phase 2: Hybrid ML+NLP Alert Service with Enhanced ML Predictions
"""

from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import select

from src.core.database import Account, AsyncSessionLocal, Tweet
from src.core.settings import get_settings
from src.ml.hybrid_scorer_v2 import HybridScorerV2
from src.tracking.trade_tracker import TradeTracker
from .telegram_bot import TelegramBot
from .formatter_ml import format_alert_with_ml


class AlertServiceMLHybridV2:
    """
    Phase 2 alert service using enhanced ML predictions.

    Threshold: 0.45 (improved model allows lower threshold)
    Weighting: 65% NLP confidence + 35% ML predictions (v2 model)
    """

    def __init__(
        self,
        bot: TelegramBot | None = None,
        interval_seconds: int = 30,
        min_hybrid_score: float = 0.45,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._interval = interval_seconds
        self._running = False
        self._min_hybrid_score = min_hybrid_score
        self._alerted_tweets = set()
        self._scorer = HybridScorerV2(nlp_weight=0.65, ml_weight=0.35)
        self._tracker = TradeTracker()

    async def check_and_send(self) -> dict:
        """
        Check eligible tweets and send alerts with Phase 2 hybrid scoring.

        Returns:
            {
                "sent": int,  # Alerts sent
                "evaluated": int,  # Tweets evaluated
                "filtered_by_score": int,  # Filtered by low hybrid score
                "profitable_predictions": int,  # ML predicted profitable
            }
        """
        threshold = self._settings.min_confidence_score
        sent = 0
        filtered = 0
        evaluated = 0
        profitable = 0

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

                # Prepare tweet data with all Phase 2 features
                tweet_data = {
                    "confidence": tweet.confidence or 0.0,
                    "sentiment": tweet.sentiment or 0.0,
                    "urgency_score": tweet.urgency_score or 0.0,
                    "like_count": tweet.like_count or 0,
                    "retweet_count": tweet.retweet_count or 0,
                    "reply_count": tweet.reply_count or 0,
                    "call_type": tweet.call_type or "long",
                    "tweet_age_hours": 0,  # Approximation (would need tweet timestamp)
                    "price_change_4h": 0.0,  # From market data
                    "price_change_1h": 0.0,  # From market data
                }

                account_data = {
                    "reliability_score": account.reliability_score or 0.5,
                    "win_rate": account.win_rate or 0.5,
                    "account_age_days": 365,  # Default
                    "avg_roi": account.avg_roi or 0.0,
                }

                score_result = self._scorer.score(tweet_data, account_data)
                hybrid_score = score_result["hybrid_score"]

                if score_result["is_profitable"]:
                    profitable += 1

                # Log alert to tracker (for dashboard)
                alert_id = f"alert_{tweet.id}_{int(tweet.created_at.timestamp())}"
                self._tracker.log_alert(
                    alert_id=alert_id,
                    username=account.username,
                    ticker=tweet.tickers[:20] if tweet.tickers else "UNKNOWN",
                    nlp_score=score_result["nlp_score"],
                    ml_score=score_result["ml_score"],
                    hybrid_score=hybrid_score,
                    ml_profitable=score_result["is_profitable"],
                    account_reliability=account.reliability_score or 0.5,
                    account_win_rate=account.win_rate or 0.5,
                    message=f"@{account.username} | Score: {hybrid_score:.2f}",
                )

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
                        "ALERT SENT (V2): @{} | Hybrid={:.3f} (NLP={:.2f}, ML={:.2f}, Profitable={}) | Conf={:.2f}",
                        account.username,
                        hybrid_score,
                        score_result["nlp_score"],
                        score_result["ml_score"],
                        score_result["is_profitable"],
                        score_result["confidence"],
                    )
                else:
                    logger.info(
                        "Alert logged (V2): @{} | Hybrid={:.3f}",
                        account.username,
                        hybrid_score,
                    )

        return {
            "sent": sent,
            "evaluated": evaluated,
            "filtered_by_score": filtered,
            "profitable_predictions": profitable,
        }

    async def run(self) -> None:
        """Main alert loop."""
        self._running = True
        await self._bot.start()
        logger.info(
            "AlertServiceMLHybridV2 started (Phase 2: min_score={:.2f}, weights: 65% NLP + 35% ML v2)",
            self._min_hybrid_score,
        )

        while self._running:
            try:
                result = await self.check_and_send()
                if result["sent"]:
                    logger.info(
                        "{} alert(s) sent (evaluated: {}, filtered: {}, profitable: {})",
                        result["sent"],
                        result["evaluated"],
                        result["filtered_by_score"],
                        result["profitable_predictions"],
                    )
            except Exception as e:
                logger.exception("Error in alert loop: {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("AlertServiceMLHybridV2 stopped.")
