"""
nlp/analyzer.py
────────────────
Analyse complète d'un tweet : combine ticker, classification, sentiment, prix.
Sans dépendance lourde (pas de transformers/torch requis).
"""

from __future__ import annotations

from dataclasses import dataclass

from .call_classifier import CallClassifier, CallType
from .price_extractor import PriceExtractor
from .ticker_extractor import TickerExtractor

# Mots bullish/bearish pour le sentiment lexical
_BULL_WORDS: frozenset[str] = frozenset({
    "bull", "bullish", "long", "buy", "moon", "pump", "green", "up",
    "breakout", "rip", "skyrocket", "support", "upside", "target",
    "accumulate", "dca", "bounce", "load", "calls", "longs",
    "🚀", "📈", "🟢", "💚", "🐂", "💪",
})
_BEAR_WORDS: frozenset[str] = frozenset({
    "bear", "bearish", "short", "sell", "dump", "red", "down",
    "crash", "drop", "breakdown", "resistance", "downside", "exit",
    "correction", "collapse", "falling", "puts", "shorts",
    "📉", "🔴", "🐻", "💔",
})


@dataclass(frozen=True)
class NLPResult:
    tickers: list[str]
    call_type: CallType
    sentiment: float       # -1.0 à 1.0
    confidence: float      # 0.0 à 1.0 (fiabilité de la classification)
    target_price: float | None
    stop_loss: float | None
    urgency_score: float   # 0.0 à 1.0


class TweetAnalyzer:
    """
    Analyse un tweet et retourne un NLPResult.
    Instancier une fois et réutiliser (les extracteurs sont stateless).
    """

    def __init__(self) -> None:
        self._ticker = TickerExtractor()
        self._classifier = CallClassifier()
        self._price = PriceExtractor()

    def analyze(self, text: str) -> NLPResult:
        tickers = self._ticker.extract(text)
        call_type, confidence = self._classifier.classify(text)
        target = self._price.extract_target_price(text)
        sl = self._price.extract_stop_loss(text)
        urgency = self._price.extract_urgency(text)
        sentiment = self._compute_sentiment(text)

        # Sans ticker, le signal est difficilement actionnable → pénalité
        if not tickers:
            confidence = round(confidence * 0.6, 3)

        return NLPResult(
            tickers=tickers,
            call_type=call_type,
            sentiment=round(sentiment, 3),
            confidence=round(confidence, 3),
            target_price=target,
            stop_loss=sl,
            urgency_score=urgency,
        )

    def _compute_sentiment(self, text: str) -> float:
        """Score de sentiment lexical ∈ [-1.0, 1.0]."""
        tokens = set(text.lower().split())
        bull = len(tokens & _BULL_WORDS)
        bear = len(tokens & _BEAR_WORDS)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total
