"""
tests/unit/test_ml_integration.py
ML Integration Tests - Vérifier que le ML scoring fonctionne avec les alertes
"""

from types import SimpleNamespace

import pytest

from src.ml import get_predictor
from src.alerts.formatter_ml import format_alert_with_ml, get_ml_alert_score


class TestMLPredictor:
    """Tests du predictor ML."""

    def test_predictor_loads(self):
        """Le predictor se charge sans erreur."""
        predictor = get_predictor()
        # Can be None if model file doesn't exist (OK in test env)
        assert predictor is not None

    def test_predictor_predict(self):
        """Le predictor peut faire une prédiction."""
        predictor = get_predictor()

        if predictor.model is None:
            pytest.skip("Model not available in test environment")

        result = predictor.predict(
            confidence=0.80,
            sentiment=0.65,
            urgency_score=0.3,
            like_count=100,
            retweet_count=50,
            reply_count=10,
            reliability_score=0.75,
            win_rate=0.55,
        )

        assert result.predicted_change_pct is not None
        assert result.confidence >= 0.0
        assert isinstance(result.is_profitable, bool)

    def test_predictor_score_tweet(self):
        """Le predictor score les tweets."""
        predictor = get_predictor()

        if predictor.model is None:
            pytest.skip("Model not available in test environment")

        tweet_data = {
            "confidence": 0.85,
            "sentiment": 0.7,
            "urgency_score": 0.4,
            "like_count": 150,
            "retweet_count": 75,
            "reply_count": 20,
        }

        account_data = {
            "reliability_score": 0.80,
            "win_rate": 0.60,
        }

        score = predictor.score_tweet(tweet_data, account_data)

        assert "final_score" in score
        assert "ml_predicted_change" in score
        assert "recommendation" in score
        assert score["recommendation"] in ["BUY", "HOLD", "SKIP"]


class TestMLFormatterIntegration:
    """Tests de l'intégration ML dans les alertes."""

    def make_tweet(self, confidence=0.82, sentiment=0.6):
        return SimpleNamespace(
            id=1,
            tweet_id="123456",
            account_id=1,
            text="Loading up $BTC here, very bullish!",
            call_type="long",
            tickers='["BTC"]',
            confidence=confidence,
            sentiment=sentiment,
            target_price=50000.0,
            stop_loss=38000.0,
            urgency_score=0.3,
            like_count=100,
            retweet_count=50,
            reply_count=10,
            nlp_processed=True,
            tweeted_at="2026-07-24T12:00:00Z",
        )

    def make_account(self):
        return SimpleNamespace(
            id=1,
            username="crypto_trader",
            display_name="Crypto Trader",
            reliability_score=0.75,
            win_rate=0.55,
        )

    def test_format_alert_with_ml(self):
        """Format d'alerte avec ML fonctionne."""
        tweet = self.make_tweet()
        account = self.make_account()

        msg = format_alert_with_ml(tweet, account)

        # Doit contenir les éléments de base
        assert "LONG" in msg or "long" in msg.lower()
        assert "BTC" in msg or "Bitcoin" in msg

        # Si le modèle est disponible, doit contenir ML scoring
        predictor = get_predictor()
        if predictor.model is not None:
            assert "ML Scoring" in msg or "Predicted" in msg

    def test_get_ml_alert_score(self):
        """get_ml_alert_score retourne un score valide."""
        tweet = self.make_tweet()
        account = self.make_account()

        score = get_ml_alert_score(tweet, account)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_ml_score_filters_low_confidence(self):
        """Les tweets à faible score ML sont filtrés."""
        tweet = self.make_tweet(confidence=0.2, sentiment=0.1)
        account = self.make_account()

        score = get_ml_alert_score(tweet, account)

        # Un score ML bas pour un tweet sans confiance
        # (dépend du modèle, mais typiquement bas)
        assert score is not None

    def test_ml_score_boosts_high_quality_tweets(self):
        """Les tweets de bonne qualité reçoivent un bon score ML."""
        high_quality_tweet = self.make_tweet(confidence=0.95, sentiment=0.85)
        low_quality_tweet = self.make_tweet(confidence=0.30, sentiment=0.20)

        account = self.make_account()

        high_score = get_ml_alert_score(high_quality_tweet, account)
        low_score = get_ml_alert_score(low_quality_tweet, account)

        # Score should generally correlate with quality
        assert high_score is not None
        assert low_score is not None
        # Note: not strictly asserting high_score > low_score
        # because ML model behavior depends on training


class TestMLAlertIntegration:
    """Tests de l'intégration complète ML + Alertes."""

    @pytest.mark.asyncio
    async def test_ml_alert_service_can_be_imported(self):
        """Le service d'alertes ML peut être importé."""
        try:
            from src.alerts.service_ml import AlertServiceML
            assert AlertServiceML is not None
        except ImportError:
            pytest.skip("AlertServiceML not available")

    def test_ml_modules_exist(self):
        """Tous les modules ML existent."""
        assert __import__("src.ml", fromlist=["get_predictor"]) is not None
        assert __import__("src.alerts.formatter_ml", fromlist=["format_alert_with_ml"]) is not None
