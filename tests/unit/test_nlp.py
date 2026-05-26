"""
tests/unit/test_nlp.py
───────────────────────
Tests unitaires du module NLP.
Aucune dépendance externe, aucune DB requise.
"""

from __future__ import annotations

import pytest

from src.nlp.ticker_extractor import TickerExtractor
from src.nlp.call_classifier import CallClassifier
from src.nlp.price_extractor import PriceExtractor
from src.nlp.analyzer import TweetAnalyzer


# ── TickerExtractor ────────────────────────────────────────────────────────────

class TestTickerExtractor:
    def setup_method(self):
        self.extractor = TickerExtractor()

    def test_dollar_ticker(self):
        result = self.extractor.extract("I'm buying $BTC today")
        assert "BTC" in result

    def test_dollar_ticker_stock(self):
        result = self.extractor.extract("$AAPL is breaking out")
        assert "AAPL" in result

    def test_pair_usdt(self):
        result = self.extractor.extract("ETH/USDT looking good")
        assert "ETH" in result

    def test_crypto_name(self):
        result = self.extractor.extract("Bitcoin is pumping hard right now")
        assert "BTC" in result

    def test_ethereum_name(self):
        result = self.extractor.extract("ethereum is going to the moon")
        assert "ETH" in result

    def test_multiple_tickers(self):
        result = self.extractor.extract("Long $BTC and $ETH, watching $NVDA too")
        assert "BTC" in result
        assert "ETH" in result
        assert "NVDA" in result

    def test_blacklisted_words_excluded(self):
        result = self.extractor.extract("I AM GOING UP NOW TO GET ALL THIS")
        assert "I" not in result
        assert "AM" not in result
        assert "UP" not in result

    def test_empty_text(self):
        result = self.extractor.extract("")
        assert result == []

    def test_no_ticker(self):
        result = self.extractor.extract("Great day today, weather is nice")
        assert result == []

    def test_returns_sorted(self):
        result = self.extractor.extract("$ETH and $BTC and $AAPL")
        assert result == sorted(result)


# ── CallClassifier ─────────────────────────────────────────────────────────────

class TestCallClassifier:
    def setup_method(self):
        self.clf = CallClassifier()

    def test_long_signal(self):
        call_type, conf = self.clf.classify("I'm going long on BTC, very bullish")
        assert call_type == "long"
        assert conf > 0.3

    def test_short_signal(self):
        call_type, conf = self.clf.classify("Shorting ETH here, bearish breakdown")
        assert call_type == "short"
        assert conf > 0.3

    def test_hold_signal(self):
        call_type, conf = self.clf.classify("Just holding, watching and waiting patiently")
        assert call_type == "hold"

    def test_info_signal(self):
        call_type, conf = self.clf.classify("BREAKING: Fed raises rates by 25bps, CPI data tomorrow")
        assert call_type == "info"

    def test_unclear_ambiguous(self):
        call_type, _ = self.clf.classify("gm everyone")
        assert call_type == "unclear"

    def test_conflict_long_short_is_unclear(self):
        call_type, _ = self.clf.classify("bullish long but also bearish short")
        assert call_type == "unclear"

    def test_confidence_range(self):
        for text in [
            "buying bitcoin now 🚀",
            "dumping everything, crash incoming 📉",
            "just watching",
            "CPI data released",
            "gm",
        ]:
            _, conf = self.clf.classify(text)
            assert 0.0 <= conf <= 1.0, f"conf={conf} out of range for: {text}"


# ── PriceExtractor ─────────────────────────────────────────────────────────────

class TestPriceExtractor:
    def setup_method(self):
        self.extractor = PriceExtractor()

    def test_target_price_dollar(self):
        result = self.extractor.extract_target_price("Target $50,000 for BTC")
        assert result == 50_000.0

    def test_target_price_k(self):
        result = self.extractor.extract_target_price("TP 45k")
        assert result == 45_000.0

    def test_target_price_tp(self):
        result = self.extractor.extract_target_price("take profit: $120")
        assert result == 120.0

    def test_stop_loss(self):
        result = self.extractor.extract_stop_loss("SL $38,000")
        assert result == 38_000.0

    def test_stop_loss_k(self):
        result = self.extractor.extract_stop_loss("stop 38k")
        assert result == 38_000.0

    def test_no_target(self):
        result = self.extractor.extract_target_price("Bitcoin is going up")
        assert result is None

    def test_no_sl(self):
        result = self.extractor.extract_stop_loss("I'm buying BTC")
        assert result is None

    def test_urgency_exclamation(self):
        score = self.extractor.extract_urgency("BTC is pumping!!!")
        assert score > 0.0

    def test_urgency_breaking(self):
        score = self.extractor.extract_urgency("BREAKING: market crash 🚨")
        assert score > 0.0

    def test_urgency_range(self):
        for text in ["gm", "BUY NOW URGENT!!! 🚨🚨", "watching patiently"]:
            score = self.extractor.extract_urgency(text)
            assert 0.0 <= score <= 1.0


# ── TweetAnalyzer ──────────────────────────────────────────────────────────────

class TestTweetAnalyzer:
    def setup_method(self):
        self.analyzer = TweetAnalyzer()

    def test_full_analysis_bullish(self):
        text = "Loading up $BTC here, target $50k, SL $38k. Very bullish! 🚀"
        result = self.analyzer.analyze(text)
        assert "BTC" in result.tickers
        assert result.call_type == "long"
        assert result.sentiment > 0
        assert result.target_price == 50_000.0
        assert result.stop_loss == 38_000.0
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.urgency_score <= 1.0

    def test_full_analysis_bearish(self):
        text = "Shorting $ETH, breakdown confirmed. Bear market. 📉"
        result = self.analyzer.analyze(text)
        assert "ETH" in result.tickers
        assert result.call_type == "short"
        assert result.sentiment < 0

    def test_no_ticker_reduces_confidence(self):
        text_with = "$BTC is bullish"
        text_without = "market is bullish"
        r_with = self.analyzer.analyze(text_with)
        r_without = self.analyzer.analyze(text_without)
        assert r_with.confidence > r_without.confidence

    def test_result_is_immutable(self):
        result = self.analyzer.analyze("$BTC moon 🚀")
        with pytest.raises(Exception):
            result.tickers = []  # type: ignore[misc]

    def test_all_fields_present(self):
        result = self.analyzer.analyze("gm")
        assert hasattr(result, "tickers")
        assert hasattr(result, "call_type")
        assert hasattr(result, "sentiment")
        assert hasattr(result, "confidence")
        assert hasattr(result, "target_price")
        assert hasattr(result, "stop_loss")
        assert hasattr(result, "urgency_score")
