"""
ml/trainer.py
──────────────
Entraîne le modèle XGBoost sur les prédictions déjà résolues (actual_profitable
non-null) et persiste le pipeline entraîné sur disque.

Opération manuelle (--ml-train), pas partie de la boucle continue —
l'entraînement est peu fréquent et délibéré, contrairement au scoring/labeling
qui tournent en continu via MLScoringService.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select

from ..core.database import Account, AsyncSessionLocal, MarketSnapshot, Prediction, Tweet
from ..core.settings import MODELS_DIR, get_settings
from .features import FEATURE_NAMES, build_features, get_primary_snapshot, to_vector
from .scorer import MODEL_FILENAME


@dataclass(frozen=True)
class TrainResult:
    version: str
    n_samples: int
    train_accuracy: float


class ModelTrainer:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def _load_training_data(self) -> tuple[list[list[float]], list[int]]:
        """Reconstruit les mêmes vecteurs de features que scorer.py, à partir
        des calls déjà résolus (issus de OutcomeLabeler)."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet, Account, Prediction)
                .join(Account, Tweet.account_id == Account.id)
                .join(Prediction, Prediction.tweet_id == Tweet.id)
                .where(Prediction.actual_profitable.isnot(None))
            )
            rows = result.all()

            vectors: list[list[float]] = []
            labels: list[int] = []

            for tweet, account, prediction in rows:
                snap_result = await session.execute(
                    select(MarketSnapshot).where(MarketSnapshot.tweet_id == tweet.id)
                )
                snapshots = snap_result.scalars().all()
                snapshot = get_primary_snapshot(tweet, snapshots)

                features = build_features(tweet, account, snapshot)
                vectors.append(to_vector(features))
                labels.append(1 if prediction.actual_profitable else 0)

        return vectors, labels

    async def train(self) -> TrainResult | None:
        vectors, labels = await self._load_training_data()
        n_samples = len(vectors)

        if n_samples < self._settings.ml_min_training_samples:
            logger.info(
                "Entraînement ML annulé : {} échantillons < minimum requis ({}).",
                n_samples, self._settings.ml_min_training_samples,
            )
            return None

        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            vectors, labels, test_size=0.2, random_state=42,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                eval_metric="logloss",
            )),
        ])
        pipeline.fit(X_train, y_train)
        train_accuracy = float(pipeline.score(X_test, y_test))

        version = f"xgboost-{datetime.now(tz=timezone.utc):%Y%m%d-%H%M%S}"

        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "version": version,
                "feature_names": FEATURE_NAMES,
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "n_samples": n_samples,
            },
            MODELS_DIR / MODEL_FILENAME,
        )

        logger.info(
            "Modèle ML entraîné : {} ({} échantillons, accuracy={:.2%}).",
            version, n_samples, train_accuracy,
        )
        return TrainResult(version=version, n_samples=n_samples, train_accuracy=train_accuracy)
