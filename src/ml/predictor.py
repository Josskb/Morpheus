"""
ML Predictor Service
Utilise le modèle XGBoost entraîné pour scorer les tweets
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger


class PredictionResult(NamedTuple):
    """Résultat de prédiction ML."""
    predicted_change_pct: float
    confidence: float
    is_profitable: bool


class MLPredictor:
    """Charge et utilise le modèle XGBoost pour faire des prédictions."""

    def __init__(self, model_path: str | Path = None):
        """
        Args:
            model_path: Chemin vers le modèle PKL. Si None, recherche dans models/
        """
        if model_path is None:
            model_dir = Path(__file__).parent.parent.parent / "models"
            model_path = model_dir / "xgboost_price_predictor_v1.pkl"

        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_names = None

        if self.model_path.exists():
            self.load_model()
        else:
            logger.warning(f"Model not found: {self.model_path}")

    def load_model(self) -> None:
        """Charge le modèle sauvegardé."""
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["features"]

            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(
        self,
        confidence: float,
        sentiment: float = 0.0,
        urgency_score: float = 0.0,
        like_count: int = 0,
        retweet_count: int = 0,
        reply_count: int = 0,
        reliability_score: float = 0.5,
        win_rate: float = 0.5,
    ) -> PredictionResult:
        """
        Prédit le changement de prix 24h.

        Args:
            confidence: Score de confiance NLP (0-1)
            sentiment: Score de sentiment (0-1)
            urgency_score: Score d'urgence (0-1)
            like_count: Nombre de likes
            retweet_count: Nombre de retweets
            reply_count: Nombre de réponses
            reliability_score: Score de fiabilité du compte
            win_rate: Taux de win du compte

        Returns:
            PredictionResult avec changement de prix prédit et confiance
        """
        if self.model is None:
            logger.warning("Model not loaded, returning neutral prediction")
            return PredictionResult(predicted_change_pct=0.0, confidence=0.0, is_profitable=False)

        # Construire le vecteur de features
        features = np.array([
            confidence,
            sentiment,
            urgency_score,
            like_count,
            retweet_count,
            reply_count,
            reliability_score,
            win_rate,
        ]).reshape(1, -1)

        # Normaliser
        features_scaled = self.scaler.transform(features)

        # Prédire
        predicted_change = float(self.model.predict(features_scaled)[0])

        # Calculer la confiance (abs du changement prédit)
        pred_confidence = min(abs(predicted_change) / 10.0, 1.0)  # Normalize

        # Profitable si changement positif
        is_profitable = predicted_change > 0.5

        return PredictionResult(
            predicted_change_pct=predicted_change,
            confidence=pred_confidence,
            is_profitable=is_profitable,
        )

    def score_tweet(
        self,
        tweet_data: dict,
        account_data: dict = None,
    ) -> dict:
        """
        Score complet d'un tweet avec ML.

        Args:
            tweet_data: Données du tweet (confidence, sentiment, etc.)
            account_data: Données du compte (reliability_score, win_rate)

        Returns:
            Dict avec prédiction ML et score final
        """
        if account_data is None:
            account_data = {}

        try:
            prediction = self.predict(
                confidence=tweet_data.get("confidence", 0.0),
                sentiment=tweet_data.get("sentiment", 0.0),
                urgency_score=tweet_data.get("urgency_score", 0.0),
                like_count=tweet_data.get("like_count", 0),
                retweet_count=tweet_data.get("retweet_count", 0),
                reply_count=tweet_data.get("reply_count", 0),
                reliability_score=account_data.get("reliability_score", 0.5),
                win_rate=account_data.get("win_rate", 0.5),
            )

            # Score final: moyenne entre NLP confidence et ML prediction
            nlp_confidence = tweet_data.get("confidence", 0.0)
            final_score = (nlp_confidence + prediction.confidence) / 2.0

            return {
                "nlp_confidence": nlp_confidence,
                "ml_predicted_change": prediction.predicted_change_pct,
                "ml_confidence": prediction.confidence,
                "ml_is_profitable": prediction.is_profitable,
                "final_score": final_score,
                "recommendation": "BUY" if prediction.is_profitable else "HOLD",
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "final_score": 0.0,
                "recommendation": "SKIP",
            }


# Singleton instance
_predictor = None


def get_predictor() -> MLPredictor:
    """Obtient l'instance singleton du prédicateur ML."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
