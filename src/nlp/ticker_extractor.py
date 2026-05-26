"""
nlp/ticker_extractor.py
────────────────────────
Extraction des tickers financiers depuis le texte d'un tweet.
Stratégie multi-couches : $TICKER > PAIR/USDT > noms courants > tickers stock connus.
"""

from __future__ import annotations

import re

# Mapping noms communs → ticker canonique
CRYPTO_NAMES: dict[str, str] = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH", "ether": "ETH",
    "solana": "SOL", "sol": "SOL",
    "binance": "BNB", "bnb": "BNB",
    "ripple": "XRP", "xrp": "XRP",
    "dogecoin": "DOGE", "doge": "DOGE",
    "cardano": "ADA", "ada": "ADA",
    "avalanche": "AVAX", "avax": "AVAX",
    "polygon": "MATIC", "matic": "MATIC",
    "polkadot": "DOT", "dot": "DOT",
    "chainlink": "LINK",
    "uniswap": "UNI",
    "cosmos": "ATOM", "atom": "ATOM",
    "litecoin": "LTC", "ltc": "LTC",
    "near": "NEAR",
    "arbitrum": "ARB", "arb": "ARB",
    "optimism": "OP",
    "injective": "INJ", "inj": "INJ",
    "sui": "SUI",
    "celestia": "TIA", "tia": "TIA",
    "sei": "SEI",
    "pepe": "PEPE",
    "shiba": "SHIB", "shib": "SHIB",
}

# Tickers actions connus pour éviter les faux positifs sur les majuscules génériques
KNOWN_STOCKS: frozenset[str] = frozenset({
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "GOOG", "AMZN",
    "SPY", "QQQ", "VIX", "AMD", "INTC", "NFLX", "UBER", "LYFT",
    "COIN", "MSTR", "HOOD", "PLTR", "SOFI", "RIVN", "LCID",
    "JPM", "GS", "BAC", "C", "WFC", "MS",
    "XOM", "CVX", "OXY",
    "GLD", "SLV", "TLT", "IWM", "DIA",
    "SHOP", "SQ", "PYPL", "V", "MA",
    "DIS", "ROKU", "SNAP", "TWTR", "PINS",
    "F", "GM", "NIO", "XPEV", "LI",
    "BABA", "JD", "PDD", "BIDU",
    "MRNA", "PFE", "JNJ", "ABBV",
})

# Mots qui ressemblent à des tickers mais n'en sont pas
TICKER_BLACKLIST: frozenset[str] = frozenset({
    "I", "A", "THE", "FOR", "AND", "OR", "BUT", "IF", "IN", "ON", "AT",
    "TO", "UP", "DO", "SO", "MY", "WE", "US", "IT", "IS", "BE", "AS",
    "OF", "BY", "PM", "AM", "DM", "RT", "GMT", "UTC", "USD", "EUR",
    "ALL", "NEW", "OLD", "NOW", "OUT", "OFF", "GET", "GOT", "NOT", "CAN",
    "ATH", "ATL", "IMO", "LOL", "FUD", "FOMO", "DYOR", "NFA", "GL",
    "GG", "OK", "TA", "AI", "TV", "OS", "IP", "IO", "TG", "RR",
    "PT", "TP", "SL", "PNL", "ROI", "REKT", "GG", "LFG",
    "JUST", "THIS", "FROM", "INTO", "THAT", "WITH", "BEEN",
})

_DOLLAR_TICKER_RE = re.compile(r'\$([A-Z]{1,6})(?:/USDT?)?', re.IGNORECASE)
_PAIR_RE = re.compile(r'\b([A-Z]{2,6})/(?:USDT?|BTC|ETH|BNB)\b')
_BARE_UPPER_RE = re.compile(r'\b([A-Z]{2,5})\b')
_CRYPTO_NAME_RE: dict[str, re.Pattern] = {
    name: re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE)
    for name in CRYPTO_NAMES
}


class TickerExtractor:
    """Extrait les tickers financiers d'un texte de tweet."""

    def extract(self, text: str) -> list[str]:
        """Retourne la liste triée des tickers détectés (sans doublons)."""
        found: set[str] = set()

        # 1. $TICKER — pattern le plus fiable
        for m in _DOLLAR_TICKER_RE.finditer(text):
            t = m.group(1).upper()
            if t not in TICKER_BLACKLIST:
                found.add(t)

        # 2. PAIR/USDT — paires de trading explicites
        for m in _PAIR_RE.finditer(text):
            found.add(m.group(1).upper())

        # 3. Noms courants crypto (bitcoin, ethereum…)
        text_lower = text.lower()
        for name, ticker in CRYPTO_NAMES.items():
            if _CRYPTO_NAME_RE[name].search(text_lower):
                found.add(ticker)

        # 4. Tickers actions connus en majuscules
        for m in _BARE_UPPER_RE.finditer(text):
            t = m.group(1).upper()
            if t in KNOWN_STOCKS:
                found.add(t)

        return sorted(found)
