"""
ml/labeler.py
──────────────
Détermine le résultat réel (profitable ou non) des calls scorés, une fois la
fenêtre d'évaluation (24h par défaut) écoulée, et met à jour les statistiques
de fiabilité du compte correspondant.

Suit le même schéma que market/snapshot_scheduler.py::SnapshotScheduler :
une classe qui possède sa propre session DB, invoquée depuis la boucle d'un
autre service (ici MLScoringService) plutôt que d'avoir sa propre boucle.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import or_, select

from ..core.database import Account, AsyncSessionLocal, MarketSnapshot, Prediction, Tweet
from ..core.settings import get_settings
from .features import get_primary_snapshot

_BATCH_SIZE = 100
_RELIABILITY_PRIOR_ALPHA = 3.0
_RELIABILITY_PRIOR_BETA = 3.0


def _apply_outcome_to_account(account, actual_profitable: bool, directional_return: float) -> None:
    """
    Met à jour les stats de fiabilité d'un compte suite à un call résolu.
    Fonction pure (pas de DB) pour rester testable isolément.
    """
    total_calls = (account.total_calls or 0) + 1
    profitable_calls = (account.profitable_calls or 0) + (1 if actual_profitable else 0)
    old_avg_roi = account.avg_roi or 0.0

    account.total_calls = total_calls
    account.profitable_calls = profitable_calls
    account.win_rate = profitable_calls / total_calls
    account.avg_roi = old_avg_roi + (directional_return - old_avg_roi) / total_calls
    # Lissage bayésien (Beta(3,3)) : neutre (0.5) à 0 call, converge vers le
    # win_rate réel avec le volume — évite un score extrême sur 1-2 calls.
    account.reliability_score = (
        (profitable_calls + _RELIABILITY_PRIOR_ALPHA)
        / (total_calls + _RELIABILITY_PRIOR_ALPHA + _RELIABILITY_PRIOR_BETA)
    )


class OutcomeLabeler:
    """Résout les prédictions dont la fenêtre d'évaluation est écoulée."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def label_pending(self) -> int:
        window = self._settings.ml_evaluation_window
        labeled = 0

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet, Account, Prediction)
                .join(Account, Tweet.account_id == Account.id)
                .outerjoin(Prediction, Prediction.tweet_id == Tweet.id)
                .where(
                    Tweet.call_type.in_(("long", "short")),
                    Tweet.tickers.isnot(None),
                    or_(Prediction.id.is_(None), Prediction.actual_profitable.is_(None)),
                )
                .limit(_BATCH_SIZE)
            )
            candidates = result.all()

            for tweet, account, prediction in candidates:
                snap_result = await session.execute(
                    select(MarketSnapshot).where(MarketSnapshot.tweet_id == tweet.id)
                )
                snapshots = snap_result.scalars().all()
                snapshot = get_primary_snapshot(tweet, snapshots)
                if snapshot is None:
                    continue

                change = getattr(snapshot, f"change_{window}", None)
                if change is None:
                    continue

                directional_return = change if tweet.call_type == "long" else -change
                actual_profitable = directional_return > 0

                if prediction is None:
                    # Cas rare : tweet éligible jamais scoré par MLScoringService.
                    prediction = Prediction(tweet_id=tweet.id, model_version="unknown")
                    session.add(prediction)

                prediction.actual_profitable = actual_profitable
                prediction.actual_change_pct = change
                prediction.evaluation_window = window

                _apply_outcome_to_account(account, actual_profitable, directional_return)

                logger.info(
                    "Label tweet={} @{} {} → {} ({:+.2f}%)",
                    tweet.tweet_id, account.username, tweet.call_type,
                    "profitable" if actual_profitable else "perte", directional_return,
                )
                labeled += 1

            await session.commit()

        return labeled
