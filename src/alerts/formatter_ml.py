"""
Enhanced Alert Formatter with ML Scoring
Améliore le formatter d'alertes avec les scores ML
"""

from __future__ import annotations

import json

from src.ml.predictor import get_predictor
from .formatter import format_alert


def format_alert_with_ml(tweet, account) -> str:
    """
    Format d'alerte enrichi avec scoring ML.

    Ajoute:
    - Prédiction ML du changement de prix 24h
    - Confiance ML
    - Recommandation finale (BUY/HOLD/SKIP)
    """
    base_alert = format_alert(tweet, account)

    # Récupérer le prédicateur ML
    predictor = get_predictor()

    if predictor.model is None:
        # Pas de modèle disponible, retourner l'alerte de base
        return base_alert

    # Construire les données pour la prédiction
    tweet_data = {
        "confidence": tweet.confidence or 0.0,
        "sentiment": tweet.sentiment or 0.0,
        "urgency_score": tweet.urgency_score or 0.0,
        "like_count": tweet.like_count or 0,
        "retweet_count": tweet.retweet_count or 0,
        "reply_count": tweet.reply_count or 0,
    }

    account_data = {
        "reliability_score": account.reliability_score or 0.5,
        "win_rate": account.win_rate or 0.5,
    }

    # Faire la prédiction
    ml_score = predictor.score_tweet(tweet_data, account_data)

    if "error" in ml_score:
        return base_alert

    # Ajouter les informations ML à l'alerte
    ml_section = f"""

<b>🤖 ML Scoring</b>
Predicted 24h change: <b>{ml_score['ml_predicted_change']:+.2f}%</b>
ML Confidence: {ml_score['ml_confidence']:.1%}
Final Score: {ml_score['final_score']:.1%}
Recommendation: <b>{ml_score['recommendation']}</b>
"""

    return base_alert + ml_section


def get_ml_alert_score(tweet, account) -> float:
    """
    Récupère le score ML final pour filtrer les alertes.
    Retourne 0.0 si pas de modèle disponible.
    """
    predictor = get_predictor()

    if predictor.model is None:
        return 0.0

    tweet_data = {
        "confidence": tweet.confidence or 0.0,
        "sentiment": tweet.sentiment or 0.0,
        "urgency_score": tweet.urgency_score or 0.0,
        "like_count": tweet.like_count or 0,
        "retweet_count": tweet.retweet_count or 0,
        "reply_count": tweet.reply_count or 0,
    }

    account_data = {
        "reliability_score": account.reliability_score or 0.5,
        "win_rate": account.win_rate or 0.5,
    }

    ml_score = predictor.score_tweet(tweet_data, account_data)

    return ml_score.get("final_score", 0.0)
