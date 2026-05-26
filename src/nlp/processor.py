"""
nlp/processor.py
─────────────────
Service NLP : lit les tweets non traités en DB, les analyse,
met à jour les champs NLP, et déclenche les snapshots marché.
"""

from __future__ import annotations

import asyncio
import json

from loguru import logger
from sqlalchemy import select

from ..core.database import AsyncSessionLocal, Tweet
from ..market.snapshot_scheduler import SnapshotScheduler
from .analyzer import TweetAnalyzer

_BATCH_SIZE = 100


class NLPProcessorService:
    """
    Service NLP en arrière-plan.
    Poll la DB toutes les `interval_seconds` pour traiter les tweets non analysés.
    """

    def __init__(
        self,
        scheduler: SnapshotScheduler | None = None,
        interval_seconds: int = 30,
    ) -> None:
        self._analyzer = TweetAnalyzer()
        self._scheduler = scheduler
        self._interval = interval_seconds
        self._running = False

    async def process_pending(self) -> int:
        """
        Traite un batch de tweets non processés.
        Retourne le nombre de tweets traités.
        """
        processed = 0

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet)
                .where(Tweet.nlp_processed == False)  # noqa: E712
                .limit(_BATCH_SIZE)
            )
            tweets = result.scalars().all()

            for tweet in tweets:
                try:
                    nlp = self._analyzer.analyze(tweet.text)

                    tweet.tickers = json.dumps(nlp.tickers) if nlp.tickers else None
                    tweet.call_type = nlp.call_type
                    tweet.sentiment = nlp.sentiment
                    tweet.confidence = nlp.confidence
                    tweet.target_price = nlp.target_price
                    tweet.stop_loss = nlp.stop_loss
                    tweet.urgency_score = nlp.urgency_score
                    tweet.nlp_processed = True

                    if nlp.tickers and self._scheduler:
                        await self._scheduler.schedule_for_tweet(
                            tweet_db_id=tweet.id,
                            tickers=nlp.tickers,
                            tweeted_at=tweet.tweeted_at,
                        )

                    logger.info(
                        "NLP tweet={} → {} sent={:+.2f} conf={:.2f} tickers={}",
                        tweet.tweet_id,
                        nlp.call_type,
                        nlp.sentiment,
                        nlp.confidence,
                        nlp.tickers or "[]",
                    )
                    processed += 1

                except Exception as e:
                    logger.error("Erreur NLP tweet {} : {}", tweet.tweet_id, e)
                    # Marque comme traité pour éviter une boucle infinie
                    tweet.nlp_processed = True

            await session.commit()

        if processed:
            logger.info("NLP : {} tweets traités.", processed)
        return processed

    async def run(self) -> None:
        """Boucle principale du service NLP."""
        self._running = True
        logger.info("NLPProcessorService démarré (intervalle={}s).", self._interval)

        while self._running:
            try:
                await self.process_pending()
            except Exception as e:
                logger.exception("Erreur dans la boucle NLP : {}", e)
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        logger.info("NLPProcessorService arrêté.")
