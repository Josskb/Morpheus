"""
tests/unit/test_ml.py
───────────────────────
Tests unitaires du module ML (features, scorer, labeler, trainer, service).
Logique pure testée avec des SimpleNamespace (pas de DB) ; labeler/trainer/
service testés avec la fixture test_db (DB SQLite temporaire).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from src.core.database import Account, MarketSnapshot, Prediction, Tweet
from src.ml.features import build_features, get_primary_snapshot, primary_ticker, to_vector, FEATURE_NAMES
from src.ml.labeler import OutcomeLabeler, _apply_outcome_to_account
from src.ml.scorer import HEURISTIC_VERSION, MLScorer
from src.ml.service import MLScoringService
from src.ml.trainer import ModelTrainer


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tweet(
    call_type: str = "long",
    tickers: list[str] | None = ("BTC",),
    sentiment: float = 0.6,
    confidence: float = 0.8,
    urgency_score: float = 0.3,
    target_price: float | None = 50_000.0,
    stop_loss: float | None = None,
    like_count: int = 100,
    retweet_count: int = 20,
    reply_count: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        tweet_id="T1",
        call_type=call_type,
        tickers=json.dumps(list(tickers)) if tickers is not None else None,
        sentiment=sentiment,
        confidence=confidence,
        urgency_score=urgency_score,
        target_price=target_price,
        stop_loss=stop_loss,
        like_count=like_count,
        retweet_count=retweet_count,
        reply_count=reply_count,
    )


def make_account(
    reliability_score: float | None = None,
    win_rate: float | None = None,
    avg_roi: float | None = None,
    total_calls: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        username="trader",
        reliability_score=reliability_score,
        win_rate=win_rate,
        avg_roi=avg_roi,
        total_calls=total_calls,
    )


def make_snapshot(
    ticker: str = "BTC",
    market_type: str = "crypto",
    price_at_tweet: float | None = 40_000.0,
    volume_at_tweet: float | None = 1_000_000.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        market_type=market_type,
        price_at_tweet=price_at_tweet,
        volume_at_tweet=volume_at_tweet,
    )


# ── features.py ───────────────────────────────────────────────────────────────

class TestFeatures:
    def test_build_features_long_tweet(self):
        features = build_features(make_tweet(call_type="long"), make_account(), make_snapshot())
        assert features["is_long"] == 1.0
        assert features["is_short"] == 0.0

    def test_build_features_short_tweet(self):
        features = build_features(make_tweet(call_type="short"), make_account(), make_snapshot())
        assert features["is_long"] == 0.0
        assert features["is_short"] == 1.0

    def test_build_features_defaults_for_new_account(self):
        account = make_account(reliability_score=None, win_rate=None, avg_roi=None, total_calls=None)
        features = build_features(make_tweet(), account, make_snapshot())
        assert features["account_reliability_score"] == 0.5
        assert features["account_win_rate"] == 0.5
        assert features["account_avg_roi"] == 0.0
        assert features["log_account_total_calls"] == 0.0

    def test_build_features_uses_existing_account_stats(self):
        account = make_account(reliability_score=0.8, win_rate=0.7, avg_roi=3.5, total_calls=10)
        features = build_features(make_tweet(), account, make_snapshot())
        assert features["account_reliability_score"] == 0.8
        assert features["account_win_rate"] == 0.7
        assert features["account_avg_roi"] == 3.5

    def test_build_features_no_snapshot(self):
        features = build_features(make_tweet(), make_account(), None)
        assert features["is_crypto"] == 0.0
        assert features["log_price_at_tweet"] == 0.0
        assert features["log_volume_at_tweet"] == 0.0

    def test_build_features_never_leaks_future_price_fields(self):
        # Le vecteur ne doit contenir que des features connues à t=0.
        assert not any("change_" in name or "price_1h" in name or "price_4h" in name
                        or "price_24h" in name or "price_7d" in name for name in FEATURE_NAMES)

    def test_to_vector_matches_feature_names_order(self):
        features = build_features(make_tweet(), make_account(), make_snapshot())
        vector = to_vector(features)
        assert vector == [features[name] for name in FEATURE_NAMES]

    def test_primary_ticker_returns_first(self):
        tweet = make_tweet(tickers=["BTC", "ETH"])
        assert primary_ticker(tweet) == "BTC"

    def test_primary_ticker_none_without_tickers(self):
        tweet = make_tweet(tickers=None)
        assert primary_ticker(tweet) is None

    def test_get_primary_snapshot_matches_ticker(self):
        tweet = make_tweet(tickers=["ETH"])
        snapshots = [make_snapshot(ticker="BTC"), make_snapshot(ticker="ETH")]
        snap = get_primary_snapshot(tweet, snapshots)
        assert snap.ticker == "ETH"

    def test_get_primary_snapshot_no_match_returns_none(self):
        tweet = make_tweet(tickers=["SOL"])
        snapshots = [make_snapshot(ticker="BTC")]
        assert get_primary_snapshot(tweet, snapshots) is None


# ── scorer.py (heuristique) ──────────────────────────────────────────────────

class TestScorerHeuristic:
    def _scorer(self, tmp_path):
        return MLScorer(model_path=tmp_path / "nonexistent.joblib")

    def test_no_model_file_uses_heuristic(self, tmp_path):
        result = self._scorer(tmp_path).score(build_features(make_tweet(), make_account(), make_snapshot()))
        assert result.model_version == HEURISTIC_VERSION

    def test_heuristic_score_stays_in_bounds(self, tmp_path):
        result = self._scorer(tmp_path).score(build_features(make_tweet(), make_account(), make_snapshot()))
        assert 0.0 <= result.score <= 1.0

    def test_heuristic_higher_confidence_higher_score(self, tmp_path):
        scorer = self._scorer(tmp_path)
        low = scorer.score(build_features(make_tweet(confidence=0.1), make_account(), make_snapshot()))
        high = scorer.score(build_features(make_tweet(confidence=0.9), make_account(), make_snapshot()))
        assert high.score > low.score

    def test_heuristic_higher_account_reliability_higher_score(self, tmp_path):
        scorer = self._scorer(tmp_path)
        low = scorer.score(build_features(make_tweet(), make_account(reliability_score=0.1), make_snapshot()))
        high = scorer.score(build_features(make_tweet(), make_account(reliability_score=0.9), make_snapshot()))
        assert high.score > low.score

    def test_heuristic_sentiment_aligned_to_call_direction(self, tmp_path):
        scorer = self._scorer(tmp_path)
        # Sentiment bearish sur un short = signal positif, pas négatif.
        bearish_short = scorer.score(build_features(
            make_tweet(call_type="short", sentiment=-0.8), make_account(), make_snapshot()))
        bullish_short = scorer.score(build_features(
            make_tweet(call_type="short", sentiment=0.8), make_account(), make_snapshot()))
        assert bearish_short.score > bullish_short.score

    def test_predicted_profitable_follows_threshold(self, tmp_path):
        scorer = self._scorer(tmp_path)
        result = scorer.score(build_features(make_tweet(confidence=0.95), make_account(reliability_score=0.95), make_snapshot()))
        assert result.predicted_profitable == (result.score >= 0.5)


# ── scorer.py (modèle) ───────────────────────────────────────────────────────

class TestScorerModel:
    def _fake_pipeline(self, proba: float = 0.73):
        pipeline = MagicMock()
        pipeline.predict_proba.return_value = [[1 - proba, proba]]
        return pipeline

    def test_loads_model_when_present(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake")

        fake_payload = {"pipeline": self._fake_pipeline(0.73), "version": "xgboost-test", "n_samples": 10}
        monkeypatch.setattr("joblib.load", MagicMock(return_value=fake_payload))

        scorer = MLScorer(model_path=model_path)
        result = scorer.score(build_features(make_tweet(), make_account(), make_snapshot()))

        assert result.model_version == "xgboost-test"
        assert result.score == pytest.approx(0.73)

    def test_reloads_only_when_mtime_changes(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake")

        load_mock = MagicMock(return_value={"pipeline": self._fake_pipeline(), "version": "v1", "n_samples": 1})
        monkeypatch.setattr("joblib.load", load_mock)

        scorer = MLScorer(model_path=model_path)
        features = build_features(make_tweet(), make_account(), make_snapshot())

        scorer.score(features)
        scorer.score(features)
        assert load_mock.call_count == 1

        # Touche le fichier (mtime change) → doit recharger.
        os.utime(model_path, (model_path.stat().st_mtime + 10, model_path.stat().st_mtime + 10))
        scorer.score(features)
        assert load_mock.call_count == 2

    def test_falls_back_to_heuristic_if_file_removed(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake")
        monkeypatch.setattr(
            "joblib.load",
            MagicMock(return_value={"pipeline": self._fake_pipeline(), "version": "v1", "n_samples": 1}),
        )

        scorer = MLScorer(model_path=model_path)
        scorer.score(build_features(make_tweet(), make_account(), make_snapshot()))

        model_path.unlink()
        result = scorer.score(build_features(make_tweet(), make_account(), make_snapshot()))
        assert result.model_version == HEURISTIC_VERSION


# ── labeler.py — _apply_outcome_to_account (pur) ────────────────────────────

class TestApplyOutcomeToAccount:
    def _fresh_account(self) -> SimpleNamespace:
        return SimpleNamespace(total_calls=0, profitable_calls=0, win_rate=None, avg_roi=None, reliability_score=None)

    def test_first_profitable_call(self):
        account = self._fresh_account()
        _apply_outcome_to_account(account, actual_profitable=True, directional_return=5.0)
        assert account.total_calls == 1
        assert account.profitable_calls == 1
        assert account.win_rate == 1.0
        assert account.avg_roi == 5.0
        assert account.reliability_score == pytest.approx(4 / 7)

    def test_first_losing_call(self):
        account = self._fresh_account()
        _apply_outcome_to_account(account, actual_profitable=False, directional_return=-3.0)
        assert account.total_calls == 1
        assert account.profitable_calls == 0
        assert account.win_rate == 0.0
        assert account.avg_roi == -3.0
        assert account.reliability_score == pytest.approx(3 / 7)

    def test_reliability_formula_neutral_at_zero_calls(self):
        # (profitable_calls + 3) / (total_calls + 6) à 0/0 == 0.5 : c'est le
        # point neutre que features.py utilise comme défaut pour un nouveau compte.
        assert (0 + 3) / (0 + 6) == 0.5

    def test_sequential_calls_update_running_stats(self):
        account = self._fresh_account()
        _apply_outcome_to_account(account, actual_profitable=True, directional_return=5.0)
        _apply_outcome_to_account(account, actual_profitable=False, directional_return=-3.0)

        assert account.total_calls == 2
        assert account.profitable_calls == 1
        assert account.win_rate == 0.5
        assert account.avg_roi == pytest.approx(1.0)  # (5.0 + -3.0) / 2
        assert account.reliability_score == pytest.approx(4 / 8)


# ── labeler.py — OutcomeLabeler (DB) ─────────────────────────────────────────

class TestLabeler:
    async def _insert_call(
        self,
        session_factory,
        call_type: str,
        change_24h: float | None,
        tweet_id: str = "L001",
        already_labeled: bool = False,
    ) -> None:
        async with session_factory() as session:
            account = Account(username=f"trader_{tweet_id}", markets="[]", tags="[]", priority="medium", enabled=True)
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id=tweet_id,
                account_id=account.id,
                text="test",
                tickers='["BTC"]',
                call_type=call_type,
                tweeted_at=datetime.now(tz=timezone.utc),
                nlp_processed=True,
            )
            session.add(tweet)
            await session.flush()

            snapshot = MarketSnapshot(
                tweet_id=tweet.id, ticker="BTC", market_type="crypto",
                price_at_tweet=40_000.0, change_24h=change_24h,
            )
            session.add(snapshot)

            prediction = Prediction(
                tweet_id=tweet.id, model_version="heuristic-v1",
                predicted_profitable=True, confidence=0.6,
            )
            if already_labeled:
                prediction.actual_profitable = True
                prediction.actual_change_pct = 1.0
                prediction.evaluation_window = "24h"
            session.add(prediction)

            await session.commit()

    @pytest.mark.asyncio
    async def test_long_with_positive_change_is_profitable(self, test_db):
        await self._insert_call(test_db, call_type="long", change_24h=8.0)
        count = await OutcomeLabeler().label_pending()
        assert count == 1

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert prediction.actual_profitable is True
        assert prediction.actual_change_pct == 8.0
        assert prediction.evaluation_window == "24h"

    @pytest.mark.asyncio
    async def test_long_with_negative_change_is_not_profitable(self, test_db):
        await self._insert_call(test_db, call_type="long", change_24h=-4.0)
        await OutcomeLabeler().label_pending()

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert prediction.actual_profitable is False

    @pytest.mark.asyncio
    async def test_short_with_negative_change_is_profitable(self, test_db):
        await self._insert_call(test_db, call_type="short", change_24h=-6.0)
        await OutcomeLabeler().label_pending()

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert prediction.actual_profitable is True
        assert prediction.actual_change_pct == -6.0

    @pytest.mark.asyncio
    async def test_short_with_positive_change_is_not_profitable(self, test_db):
        await self._insert_call(test_db, call_type="short", change_24h=6.0)
        await OutcomeLabeler().label_pending()

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert prediction.actual_profitable is False

    @pytest.mark.asyncio
    async def test_skips_when_window_not_yet_available(self, test_db):
        await self._insert_call(test_db, call_type="long", change_24h=None)
        count = await OutcomeLabeler().label_pending()
        assert count == 0

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert prediction.actual_profitable is None

    @pytest.mark.asyncio
    async def test_skips_already_labeled(self, test_db):
        await self._insert_call(test_db, call_type="long", change_24h=8.0, already_labeled=True)
        count = await OutcomeLabeler().label_pending()
        assert count == 0

    @pytest.mark.asyncio
    async def test_updates_account_stats(self, test_db):
        await self._insert_call(test_db, call_type="long", change_24h=8.0, tweet_id="A1")
        await OutcomeLabeler().label_pending()

        async with test_db() as session:
            account = (await session.execute(select(Account))).scalar_one()
        assert account.total_calls == 1
        assert account.profitable_calls == 1
        assert account.win_rate == 1.0
        assert account.reliability_score == pytest.approx(4 / 7)


# ── trainer.py — ModelTrainer (DB) ───────────────────────────────────────────

class TestTrainer:
    async def _insert_labeled_call(
        self, session_factory, tweet_id: str, profitable: bool, change: float,
    ) -> None:
        async with session_factory() as session:
            account = Account(username=f"trader_{tweet_id}", markets="[]", tags="[]", priority="medium", enabled=True)
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id=tweet_id, account_id=account.id, text="test",
                tickers='["BTC"]', call_type="long",
                sentiment=0.5, confidence=0.7, urgency_score=0.2,
                tweeted_at=datetime.now(tz=timezone.utc), nlp_processed=True,
            )
            session.add(tweet)
            await session.flush()

            session.add(MarketSnapshot(
                tweet_id=tweet.id, ticker="BTC", market_type="crypto", price_at_tweet=100.0,
            ))
            session.add(Prediction(
                tweet_id=tweet.id, model_version="heuristic-v1",
                actual_profitable=profitable, actual_change_pct=change, evaluation_window="24h",
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_insufficient_samples_returns_none(self, test_db):
        await self._insert_labeled_call(test_db, "TR1", True, 5.0)

        trainer = ModelTrainer()
        result = await trainer.train()
        assert result is None

    @pytest.mark.asyncio
    async def test_trains_and_persists_model(self, test_db, tmp_path, monkeypatch):
        for i in range(20):
            await self._insert_labeled_call(
                test_db, f"TR{i}", profitable=(i % 2 == 0), change=(2.0 if i % 2 == 0 else -2.0),
            )

        trainer = ModelTrainer()
        monkeypatch.setattr(trainer._settings, "ml_min_training_samples", 10)
        monkeypatch.setattr("src.ml.trainer.MODELS_DIR", tmp_path)

        result = await trainer.train()
        assert result is not None
        assert result.n_samples == 20
        assert (tmp_path / "xgboost_model.joblib").exists()

        # Un nouveau MLScorer pointant sur ce fichier utilise le modèle entraîné.
        fresh_scorer = MLScorer(model_path=tmp_path / "xgboost_model.joblib")
        score_result = fresh_scorer.score(build_features(make_tweet(), make_account(), make_snapshot()))
        assert score_result.model_version == result.version


# ── service.py — MLScoringService (DB) ───────────────────────────────────────

class TestMLScoringService:
    async def _insert_tweet(
        self, session_factory, call_type: str = "long", tickers: str | None = '["BTC"]',
        ml_score: float | None = None, nlp_processed: bool = True, tweet_id: str = "S001",
    ) -> None:
        async with session_factory() as session:
            account = Account(username=f"trader_{tweet_id}", markets="[]", tags="[]", priority="medium", enabled=True)
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id=tweet_id, account_id=account.id, text="test",
                tickers=tickers, call_type=call_type,
                sentiment=0.5, confidence=0.7, urgency_score=0.2, ml_score=ml_score,
                tweeted_at=datetime.now(tz=timezone.utc), nlp_processed=nlp_processed,
            )
            session.add(tweet)
            await session.flush()

            session.add(MarketSnapshot(
                tweet_id=tweet.id, ticker="BTC", market_type="crypto", price_at_tweet=100.0,
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_scores_eligible_tweet(self, test_db):
        await self._insert_tweet(test_db)
        count = await MLScoringService().score_pending()
        assert count == 1

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet))).scalar_one()
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert tweet.ml_score is not None
        assert 0.0 <= tweet.ml_score <= 1.0
        assert prediction.tweet_id == tweet.id

    @pytest.mark.asyncio
    async def test_skips_non_actionable_call_type(self, test_db):
        await self._insert_tweet(test_db, call_type="hold")
        count = await MLScoringService().score_pending()
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_already_scored(self, test_db):
        await self._insert_tweet(test_db, ml_score=0.5)
        count = await MLScoringService().score_pending()
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_without_tickers(self, test_db):
        await self._insert_tweet(test_db, tickers=None)
        count = await MLScoringService().score_pending()
        assert count == 0
