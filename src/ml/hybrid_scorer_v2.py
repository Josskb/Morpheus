"""
Phase 2: Hybrid ML+NLP Scoring with Enhanced ML Model
Uses engineered features v2 model for better predictions
"""

from __future__ import annotations

from src.ml.predictor_v2 import get_predictor


class HybridScorerV2:
    """Phase 2 hybrid scorer using v2 model with feature engineering."""

    def __init__(
        self,
        nlp_weight: float = 0.65,
        ml_weight: float = 0.35,
    ):
        self.nlp_weight = nlp_weight
        self.ml_weight = ml_weight
        self.predictor = get_predictor()

    def score(self, tweet_data: dict, account_data: dict | None = None) -> dict:
        """
        Phase 2 scoring with enhanced features.

        Args:
            tweet_data: Tweet info with all available fields
            account_data: Account info (reliability, win_rate, account_age_days)

        Returns:
            {
                "nlp_score": float,
                "ml_score": float,
                "hybrid_score": float,
                "is_profitable": bool,
                "confidence": float,
            }
        """
        if account_data is None:
            account_data = {}

        nlp_conf = tweet_data.get("confidence", 0.0)

        # Get ML prediction using Phase 2 model
        ml_pred = self.predictor.predict(
            confidence=nlp_conf,
            sentiment=tweet_data.get("sentiment", 0.0),
            urgency_score=tweet_data.get("urgency_score", 0.0),
            like_count=tweet_data.get("like_count", 0),
            retweet_count=tweet_data.get("retweet_count", 0),
            reply_count=tweet_data.get("reply_count", 0),
            reliability_score=account_data.get("reliability_score", 0.5),
            win_rate=account_data.get("win_rate", 0.5),
            account_age_days=account_data.get("account_age_days", 365),
            tweet_age_hours=tweet_data.get("tweet_age_hours", 0),
            call_type=tweet_data.get("call_type", "long"),
            price_change_4h=tweet_data.get("price_change_4h", 0.0),
            price_change_1h=tweet_data.get("price_change_1h", 0.0),
            avg_roi=account_data.get("avg_roi", 0.0),
        )

        # ML score from Phase 2 (confidence of profitability)
        ml_score = ml_pred.confidence

        # Hybrid score: 65% NLP (proven) + 35% ML (improved)
        hybrid = (self.nlp_weight * nlp_conf) + (self.ml_weight * ml_score)
        hybrid = min(max(hybrid, 0.0), 1.0)  # Clamp to 0-1

        return {
            "nlp_score": float(nlp_conf),
            "ml_score": float(ml_score),
            "hybrid_score": float(hybrid),
            "is_profitable": ml_pred.is_profitable,
            "confidence": float(ml_pred.confidence),
        }
