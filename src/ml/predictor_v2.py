"""
Phase 2 ML Predictor: Uses enhanced v2 model with feature engineering
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from loguru import logger


class PredictionResult(NamedTuple):
    predicted_change_pct: float
    confidence: float
    is_profitable: bool


class MLPredictorV2:
    """Phase 2 ML predictor using engineered features."""

    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgboost_v2_phase2.pkl"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.scaler = None
        self.feature_names = None

        try:
            self._load()
            self._initialized = True
            logger.info("✓ MLPredictorV2 loaded (Phase 2 with feature engineering)")
        except Exception as e:
            logger.warning(f"Failed to load Phase 2 model: {e}, predictions unavailable")
            self._initialized = True

    def _load(self):
        """Load Phase 2 model and scaler."""
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")

        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]

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
        account_age_days: int = 365,
        tweet_age_hours: float = 0,
        call_type: str = "long",
        price_change_4h: float = 0.0,
        price_change_1h: float = 0.0,
        avg_roi: float = 0.0,
    ) -> PredictionResult:
        """Predict profitability using Phase 2 model."""

        if self.model is None:
            # Fallback if model not loaded
            return PredictionResult(
                predicted_change_pct=0.0,
                confidence=0.5,
                is_profitable=confidence >= 0.6,
            )

        # Build feature vector matching Phase 2 training
        call_type_encoded = 1 if call_type == "long" else -1

        features = np.array([
            [
                confidence,
                sentiment,
                urgency_score,
                like_count,
                retweet_count,
                reply_count,
                (retweet_count + reply_count) / max(like_count + retweet_count + reply_count, 1),  # engagement_rate
                retweet_count / max(like_count, 1),  # retweet_ratio
                reliability_score,
                win_rate,
                avg_roi,
                account_age_days,
                tweet_age_hours,
                call_type_encoded,
                price_change_4h,
                price_change_1h,
            ]
        ])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0, 1]

        # Convert to result (1 = profitable, 0 = not profitable)
        is_profitable = prediction == 1
        confidence_score = probability if is_profitable else (1 - probability)

        return PredictionResult(
            predicted_change_pct=float(probability * 5.0),  # Scale to ~5% range
            confidence=float(confidence_score),
            is_profitable=is_profitable,
        )


def get_predictor() -> MLPredictorV2:
    """Get singleton predictor instance."""
    return MLPredictorV2()
