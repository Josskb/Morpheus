"""
ml/scorer.py
─────────────
Calcule le ml_score (0-1) d'un tweet à partir de son vecteur de features.

Utilise le modèle XGBoost entraîné (models/xgboost_model.joblib) s'il existe,
avec repli automatique sur une heuristique pondérée sinon (cold start : pas
encore de modèle entraîné au premier lancement). Le modèle est rechargé
automatiquement dès que son fichier change (mtime), sans redémarrage du
service — MLScoringService ne sait jamais laquelle des deux voies a produit
le score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..core.settings import MODELS_DIR, get_settings
from .features import to_vector

MODEL_FILENAME = "xgboost_model.joblib"
HEURISTIC_VERSION = "heuristic-v1"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    predicted_profitable: bool
    model_version: str


class MLScorer:
    """
    Charge paresseusement le modèle entraîné depuis le disque et le recharge
    si le fichier a changé. Sans modèle disponible, utilise une heuristique
    basée sur la confiance NLP, le sentiment aligné à la direction du call,
    l'urgence, et la fiabilité historique du compte.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._settings = get_settings()
        self._model_path = model_path or (MODELS_DIR / MODEL_FILENAME)
        self._pipeline = None
        self._model_version = HEURISTIC_VERSION
        self._loaded_mtime: float | None = None

    def _maybe_reload(self) -> None:
        try:
            mtime = self._model_path.stat().st_mtime
        except FileNotFoundError:
            if self._pipeline is not None:
                logger.warning("Modèle ML introuvable, repli sur l'heuristique.")
            self._pipeline = None
            self._model_version = HEURISTIC_VERSION
            self._loaded_mtime = None
            return

        if self._loaded_mtime == mtime:
            return

        import joblib

        payload = joblib.load(self._model_path)
        self._pipeline = payload["pipeline"]
        self._model_version = payload["version"]
        self._loaded_mtime = mtime
        logger.info("Modèle ML chargé : {} ({} échantillons)", payload["version"], payload.get("n_samples", "?"))

    def score(self, features: dict[str, float]) -> ScoreResult:
        self._maybe_reload()
        threshold = self._settings.ml_score_threshold

        if self._pipeline is not None:
            vector = [to_vector(features)]
            proba = float(self._pipeline.predict_proba(vector)[0][1])
            return ScoreResult(proba, proba >= threshold, self._model_version)

        score = self._heuristic(features)
        return ScoreResult(score, score >= threshold, HEURISTIC_VERSION)

    @staticmethod
    def _heuristic(features: dict[str, float]) -> float:
        # Sentiment aligné à la direction du call : un sentiment bearish sur un
        # short est un signal positif, pas négatif.
        raw_sentiment = features["sentiment"]
        aligned_sentiment = raw_sentiment if features["is_long"] else -raw_sentiment
        norm_sentiment = (aligned_sentiment + 1) / 2

        score = (
            0.30 * features["confidence"]
            + 0.25 * norm_sentiment
            + 0.15 * features["urgency_score"]
            + 0.30 * features["account_reliability_score"]
        )
        return min(1.0, max(0.0, score))
