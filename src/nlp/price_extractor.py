"""
nlp/price_extractor.py
───────────────────────
Extraction du prix cible, stop loss, et score d'urgence depuis le texte.
"""

from __future__ import annotations

import re

# Prix cible : "target $45k", "TP 45,000", "PT $120", "take profit 0.85"
_TARGET_RE = re.compile(
    r'(?:target|tp\d?|take\s*profit|pt|price\s*target)\s*:?\s*'
    r'\$?([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
    re.IGNORECASE,
)

# Stop loss : "SL $38k", "stop 38000", "stop loss: $38,500"
_SL_RE = re.compile(
    r'(?:sl|stop\s*loss|stop)\s*:?\s*\$?([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
    re.IGNORECASE,
)

# Signaux d'urgence
_URGENCY_RE = re.compile(
    r'(!{2,}|URGENT|BREAKING|ALERT|NOW|ASAP|EMERGENCY|🚨|⚠️|🔥|‼️)',
    re.IGNORECASE,
)


def _parse_amount(num_str: str, suffix: str) -> float | None:
    """Convertit "45,000" + "k" → 45_000_000.0. Retourne None si invalide."""
    try:
        val = float(num_str.replace(",", ""))
        s = suffix.upper()
        if s == "K":
            val *= 1_000
        elif s == "M":
            val *= 1_000_000
        # Sanity check : pas de prix négatif ou astronomique
        if val <= 0 or val > 1_000_000_000:
            return None
        return val
    except ValueError:
        return None


class PriceExtractor:
    """Extrait les données de prix et l'urgence d'un tweet."""

    def extract_target_price(self, text: str) -> float | None:
        m = _TARGET_RE.search(text)
        if m:
            return _parse_amount(m.group(1), m.group(2))
        return None

    def extract_stop_loss(self, text: str) -> float | None:
        m = _SL_RE.search(text)
        if m:
            return _parse_amount(m.group(1), m.group(2))
        return None

    def extract_urgency(self, text: str) -> float:
        """Score d'urgence ∈ [0.0, 1.0]."""
        urgency_hits = len(_URGENCY_RE.findall(text))
        excl_count = text.count("!")
        score = urgency_hits * 0.3 + excl_count * 0.05
        return round(min(1.0, score), 3)
