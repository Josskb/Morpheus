"""
Hybrid ML + NLP Scoring for Alerts
Combines NLP confidence with ML predictions for better recommendations
"""

from __future__ import annotations

from src.ml.predictor import get_predictor


def compute_hybrid_score(
    nlp_confidence: float,
    sentiment: float = 0.0,
    urgency_score: float = 0.0,
    like_count: int = 0,
    retweet_count: int = 0,
    reply_count: int = 0,
    reliability_score: float = 0.5,
    win_rate: float = 0.5,
    nlp_weight: float = 0.70,
    ml_weight: float = 0.30,
) -> float:
    """
    Compute hybrid score combining NLP and ML.

    Scoring formula:
        final_score = (nlp_weight * nlp_confidence) + (ml_weight * ml_score)

    Args:
        nlp_confidence: NLP model confidence (0-1)
        sentiment: Sentiment score (0-1)
        urgency_score: Urgency score (0-1)
        like_count: Number of likes
        retweet_count: Number of retweets
        reply_count: Number of replies
        reliability_score: Account reliability (0-1)
        win_rate: Account win rate (0-1)
        nlp_weight: Weight for NLP score (default 0.70)
        ml_weight: Weight for ML score (default 0.30)

    Returns:
        Final hybrid score (0-1)
    """
    predictor = get_predictor()

    # Get ML score
    if predictor.model is not None:
        ml_pred = predictor.predict(
            confidence=nlp_confidence,
            sentiment=sentiment,
            urgency_score=urgency_score,
            like_count=like_count,
            retweet_count=retweet_count,
            reply_count=reply_count,
            reliability_score=reliability_score,
            win_rate=win_rate,
        )
        # Normalize ML confidence to 0-1 range
        ml_score = min(ml_pred.confidence, 1.0)
    else:
        ml_score = 0.5  # Neutral if model not available

    # Compute hybrid score
    hybrid_score = (nlp_weight * nlp_confidence) + (ml_weight * ml_score)

    return min(hybrid_score, 1.0)  # Ensure 0-1 range


def get_recommendation(
    hybrid_score: float,
    ml_predicted_change: float = 0.0,
    profitable_threshold: float = 0.50,
) -> str:
    """
    Get trading recommendation based on scores.

    Args:
        hybrid_score: Combined NLP+ML score (0-1)
        ml_predicted_change: Predicted price change %
        profitable_threshold: Minimum score for BUY recommendation

    Returns:
        "BUY", "HOLD", or "SKIP"
    """
    if hybrid_score >= profitable_threshold:
        if ml_predicted_change > 0.5:
            return "BUY"
        else:
            return "HOLD"
    else:
        return "SKIP"


class HybridScorer:
    """Wrapper for consistent scoring across the system."""

    def __init__(
        self,
        nlp_weight: float = 0.70,
        ml_weight: float = 0.30,
    ):
        self.nlp_weight = nlp_weight
        self.ml_weight = ml_weight
        self.predictor = get_predictor()

    def score(self, tweet_data: dict, account_data: dict = None) -> dict:
        """
        Score a tweet with hybrid approach.

        Args:
            tweet_data: Tweet info (confidence, sentiment, engagement)
            account_data: Account info (reliability_score, win_rate)

        Returns:
            {
                "nlp_score": float,
                "ml_score": float,
                "hybrid_score": float,
                "ml_predicted_change": float,
                "recommendation": str,
            }
        """
        if account_data is None:
            account_data = {}

        nlp_conf = tweet_data.get("confidence", 0.0)

        # Get ML prediction
        if self.predictor.model is not None:
            ml_pred = self.predictor.predict(
                confidence=nlp_conf,
                sentiment=tweet_data.get("sentiment", 0.0),
                urgency_score=tweet_data.get("urgency_score", 0.0),
                like_count=tweet_data.get("like_count", 0),
                retweet_count=tweet_data.get("retweet_count", 0),
                reply_count=tweet_data.get("reply_count", 0),
                reliability_score=account_data.get("reliability_score", 0.5),
                win_rate=account_data.get("win_rate", 0.5),
            )
            ml_score = min(ml_pred.confidence, 1.0)
            ml_change = ml_pred.predicted_change_pct
        else:
            ml_score = 0.5
            ml_change = 0.0

        # Hybrid score
        hybrid = (self.nlp_weight * nlp_conf) + (self.ml_weight * ml_score)
        hybrid = min(hybrid, 1.0)

        # Recommendation
        rec = get_recommendation(hybrid, ml_change)

        return {
            "nlp_score": nlp_conf,
            "ml_score": ml_score,
            "hybrid_score": hybrid,
            "ml_predicted_change": ml_change,
            "recommendation": rec,
        }
