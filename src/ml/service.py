"""
ml/service.py
──────────────
Service ML : score les tweets actionnables en attente (XGBoost ou heuristique
de repli) et déclenche le labeling des calls dont l'issue est maintenant
connue. Mirroir de NLPProcessorService / AlertService.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import select

from ..core.database import Account, AsyncSessionLocal, MarketSnapshot, Prediction, Tweet
from .features import build_features, get_primary_snapshot
from .labeler import OutcomeLabeler
from .scorer import MLScorer

_BATCH_SIZE = 100
_ACTIONABLE_CALLS = {"long", "short"}


class MLScoringService:
    """
    Boucle en arrière-plan : à chaque tick, score les tweets actionnables non
    encore scorés, puis résout les prédictions dont la fenêtre d'évaluation
    est écoulée (même intervalle — le labeling est un compagnon léger du
    scoring, pas un flux indépendant).
    """

    def __init__(
        self,
        scorer: MLScorer | None = None,
        labeler: OutcomeLabeler | None = None,
        interval_seconds: int = 60,
    ) -> None:
        self._scorer = scorer or MLScorer()
        self._labeler = labeler or OutcomeLabeler()
        self._interval = interval_seconds
        self._running = False

    async def score_pending(self) -> int:
        """Score un batch de tweets actionnables pas encore scorés. Retourne le nombre scoré."""
        scored = 0

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet, Account)
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,        # noqa: E712
                    Tweet.call_type.in_(_ACTIONABLE_CALLS),
                    Tweet.tickers.isnot(None),
                    Tweet.ml_score.is_(None),
                )
                .limit(_BATCH_SIZE)
            )
            candidates = result.all()

            for tweet, account in candidates:
                try:
                    snap_result = await session.execute(
                        select(MarketSnapshot).where(MarketSnapshot.tweet_id == tweet.id)
                    )
                    snapshots = snap_result.scalars().all()
                    snapshot = get_primary_snapshot(tweet, snapshots)

                    features = build_features(tweet, account, snapshot)
                    outcome = self._scorer.score(features)

                    tweet.ml_score = outcome.score
                    session.add(Prediction(
                        tweet_id=tweet.id,
                        model_version=outcome.model_version,
                        predicted_profitable=outcome.predicted_profitable,
                        confidence=outcome.score,
                    ))

                    logger.info(
                        "ML tweet={} @{} {} score={:.3f} ({})",
                        tweet.tweet_id, account.username, tweet.call_type,
                        outcome.score, outcome.model_version,
                    )
                    scored += 1

                except Exception as e:
                    logger.error("Erreur scoring ML tweet {} : {}", tweet.tweet_id, e)

            await session.commit()

        return scored

    async def run(self) -> None:
        self._running = True
        logger.info("MLScoringService démarré (intervalle={}s).", self._interval)

        while self._running:
            try:
                scored = await self.score_pending()
                labeled = await self._labeler.label_pending()
                if scored or labeled:
                    logger.info("ML : {} scoré(s), {} labellisé(s).", scored, labeled)
            except Exception as e:
                logger.exception("Erreur dans la boucle ML : {}", e)
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        logger.info("MLScoringService arrêté.")
