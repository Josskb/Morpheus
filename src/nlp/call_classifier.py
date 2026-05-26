"""
nlp/call_classifier.py
───────────────────────
Classifie un tweet en : long | short | hold | info | unclear
Approche : scoring par patterns regex pondérés.
"""

from __future__ import annotations

import re
from typing import Literal

CallType = Literal["long", "short", "hold", "info", "unclear"]

# Chaque entrée est (pattern, poids) — les patterns plus spécifiques ont plus de poids
_LONG_RULES: list[tuple[str, float]] = [
    (r'\b(long|longing|buy|buying|bullish|bull)\b', 1.5),
    (r'\b(accumulate|dca|load(ing)?|add(ing)?)\b', 1.2),
    (r'\b(moon|mooning|pump(ing)?|rip(ping)?|skyrocket)\b', 1.0),
    (r'\b(breakout|break\s*out|bounce|upside|uptrend)\b', 1.0),
    (r'\b(calls?\b(?!.*puts?))\b', 1.2),
    (r'🚀|📈|🟢|💚|🐂', 0.8),
]

_SHORT_RULES: list[tuple[str, float]] = [
    (r'\b(short|shorting|sell|selling|bearish|bear)\b', 1.5),
    (r'\b(dump(ing)?|crash(ing)?|drop(ping)?|fall(ing)?)\b', 1.0),
    (r'\b(resistance|downside|downtrend|rejection|breakdown)\b', 1.0),
    (r'\b(exit(ing)?|cut(ting)?|close\s+position)\b', 1.2),
    (r'\b(puts?\b(?!.*calls?))\b', 1.2),
    (r'📉|🔴|🐻|💔', 0.8),
]

_HOLD_RULES: list[tuple[str, float]] = [
    (r'\b(hold(ing)?|hodl|wait(ing)?|watching)\b', 1.5),
    (r'\b(sideways|ranging|consolidat(ing|ion)|neutral)\b', 1.2),
    (r'\b(patient|patience|dca)\b', 0.8),
]

_INFO_RULES: list[tuple[str, float]] = [
    (r'\b(breaking|just\s+in|update|report|announce)\b', 1.5),
    (r'\b(news|cpi|fomc|gdp|inflation|data|fed|rates?)\b', 1.2),
    (r'\b(thread|analysis|chart|reminder|watch)\b', 0.8),
    (r'\b(sec|fda|earnings|revenue|guidance)\b', 1.0),
]

# Compile toutes les règles
_COMPILED: dict[str, list[tuple[re.Pattern, float]]] = {
    "long": [(re.compile(p, re.IGNORECASE), w) for p, w in _LONG_RULES],
    "short": [(re.compile(p, re.IGNORECASE), w) for p, w in _SHORT_RULES],
    "hold": [(re.compile(p, re.IGNORECASE), w) for p, w in _HOLD_RULES],
    "info": [(re.compile(p, re.IGNORECASE), w) for p, w in _INFO_RULES],
}

# Poids max possible par catégorie (pour normaliser la confidence)
_MAX_SCORE: dict[str, float] = {
    k: sum(w for _, w in rules)
    for k, rules in [
        ("long", _LONG_RULES), ("short", _SHORT_RULES),
        ("hold", _HOLD_RULES), ("info", _INFO_RULES),
    ]
}


class CallClassifier:
    """Classifie un tweet en long/short/hold/info/unclear avec score de confiance."""

    def classify(self, text: str) -> tuple[CallType, float]:
        """
        Retourne (call_type, confidence).
        confidence ∈ [0.0, 1.0] — proportion du score max atteint.
        """
        scores: dict[str, float] = {k: 0.0 for k in _COMPILED}

        for call_type, rules in _COMPILED.items():
            for pattern, weight in rules:
                if pattern.search(text):
                    scores[call_type] += weight

        total = sum(scores.values())
        if total == 0.0:
            return "unclear", 0.05

        best: str = max(scores, key=lambda k: scores[k])
        best_score: float = scores[best]

        # Ambiguïté long/short → unclear
        if best in ("long", "short"):
            other = "short" if best == "long" else "long"
            if scores[other] >= best_score * 0.8:
                return "unclear", 0.15

        raw_confidence = best_score / _MAX_SCORE[best]
        # Écrase dans [0.15, 0.95] pour éviter les extrêmes
        confidence = 0.15 + raw_confidence * 0.80

        return best, round(min(0.95, confidence), 3)  # type: ignore[return-value]
