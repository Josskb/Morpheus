"""
market/fetcher.py
──────────────────
Récupération des données de marché :
- Stocks / ETFs via yfinance (pas de clé requise)
- Crypto via ccxt (Binance public endpoints, pas de clé pour lecture)

Interface unifiée : get_current_price(ticker, market_type) → float | None
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import ccxt.async_support as ccxt
import yfinance as yf
from loguru import logger
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt, wait_exponential,
)

from ..core.settings import get_settings

MarketType = Literal["stock", "crypto"]


# ── Stocks via yfinance ────────────────────────────────────────────────────────

class StockFetcher:
    """
    Récupère les prix d'actions/ETFs via yfinance.
    yfinance est synchrone → on utilise asyncio.to_thread pour ne pas bloquer.
    """

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_price(self, ticker: str) -> float | None:
        """Retourne le dernier prix disponible pour un ticker."""
        try:
            data = await asyncio.to_thread(self._fetch_price, ticker)
            return data
        except Exception as e:
            logger.warning("yfinance erreur pour {} : {}", ticker, e)
            return None

    @staticmethod
    def _fetch_price(ticker: str) -> float | None:
        t = yf.Ticker(ticker)
        # fast_info est plus rapide que history pour le prix actuel
        try:
            price = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
            if price:
                return float(price)
        except Exception:
            pass
        # Fallback : dernier close
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None

    async def get_ohlcv(
        self,
        ticker: str,
        period: str = "5d",
        interval: str = "1h",
    ) -> dict | None:
        """
        Retourne les données OHLCV sous forme de dict sérialisable.
        period : 1d, 5d, 1mo, 3mo, 6mo, 1y
        interval : 1m, 5m, 15m, 30m, 1h, 1d
        """
        try:
            df = await asyncio.to_thread(
                yf.download,
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None
            return {
                "ticker": ticker,
                "interval": interval,
                "timestamps": df.index.astype(str).tolist(),
                "open": df["Open"].tolist(),
                "high": df["High"].tolist(),
                "low": df["Low"].tolist(),
                "close": df["Close"].tolist(),
                "volume": df["Volume"].tolist(),
            }
        except Exception as e:
            logger.warning("OHLCV stock {} erreur : {}", ticker, e)
            return None


# ── Crypto via ccxt ────────────────────────────────────────────────────────────

class CryptoFetcher:
    """
    Récupère les prix crypto via ccxt (Binance par défaut).
    Endpoints publics → pas besoin de clé API pour la lecture.
    """

    def __init__(self) -> None:
        settings = get_settings()
        kwargs: dict = {"enableRateLimit": True}
        if settings.binance_api_key:
            kwargs["apiKey"] = settings.binance_api_key
            kwargs["secret"] = settings.binance_api_secret
        self._exchange = ccxt.binance(kwargs)

    async def close(self) -> None:
        await self._exchange.close()

    def _normalize_pair(self, ticker: str) -> str:
        """BTC → BTC/USDT, ETH/USDT → ETH/USDT"""
        ticker = ticker.upper().replace("$", "")
        if "/" not in ticker:
            return f"{ticker}/USDT"
        return ticker

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_price(self, ticker: str) -> float | None:
        pair = self._normalize_pair(ticker)
        try:
            ticker_data = await self._exchange.fetch_ticker(pair)
            return float(ticker_data["last"])
        except ccxt.BadSymbol:
            logger.debug("Paire crypto inconnue : {}", pair)
            return None
        except Exception as e:
            logger.warning("ccxt erreur pour {} : {}", pair, e)
            return None

    async def get_ohlcv(
        self,
        ticker: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> dict | None:
        pair = self._normalize_pair(ticker)
        try:
            raw = await self._exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if not raw:
                return None
            timestamps, opens, highs, lows, closes, volumes = zip(*raw)
            return {
                "ticker": pair,
                "interval": timeframe,
                "timestamps": [
                    datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                    for ts in timestamps
                ],
                "open": list(opens),
                "high": list(highs),
                "low": list(lows),
                "close": list(closes),
                "volume": list(volumes),
            }
        except Exception as e:
            logger.warning("OHLCV crypto {} erreur : {}", ticker, e)
            return None


# ── Façade unifiée ─────────────────────────────────────────────────────────────

class MarketDataFetcher:
    """
    Point d'entrée unique pour toutes les données de marché.
    Détecte automatiquement si un ticker est crypto ou stock.
    """

    # Préfixes courants pour détecter la crypto dans les tweets
    CRYPTO_TICKERS = {
        "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX",
        "MATIC", "DOT", "LINK", "UNI", "ATOM", "LTC", "BCH", "NEAR",
        "APT", "ARB", "OP", "INJ", "SUI", "TIA", "SEI",
    }

    def __init__(self) -> None:
        self._stock = StockFetcher()
        self._crypto = CryptoFetcher()

    async def close(self) -> None:
        await self._crypto.close()

    def detect_market_type(self, ticker: str) -> MarketType:
        """Détermine si un ticker est crypto ou stock."""
        clean = ticker.upper().replace("$", "").split("/")[0]
        return "crypto" if clean in self.CRYPTO_TICKERS else "stock"

    async def get_price(
        self,
        ticker: str,
        market_type: MarketType | None = None,
    ) -> tuple[float | None, MarketType]:
        """
        Retourne (prix, market_type).
        Si market_type non fourni, le détecte automatiquement.
        """
        mtype = market_type or self.detect_market_type(ticker)
        if mtype == "crypto":
            price = await self._crypto.get_price(ticker)
        else:
            price = await self._stock.get_price(ticker)
        return price, mtype

    async def snapshot_tweet_tickers(
        self,
        tickers: list[str],
        market_types: dict[str, MarketType] | None = None,
    ) -> dict[str, dict]:
        """
        Prend un snapshot des prix pour une liste de tickers.
        Retourne {ticker: {price, market_type, timestamp}}.
        Utilisé immédiatement après détection d'un nouveau tweet.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        results: dict[str, dict] = {}

        # Fetch en parallèle
        tasks = {
            ticker: self.get_price(
                ticker,
                market_types.get(ticker) if market_types else None,
            )
            for ticker in tickers
        }
        fetched = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for ticker, result in zip(tasks.keys(), fetched):
            if isinstance(result, Exception):
                logger.warning("Snapshot échoué pour {} : {}", ticker, result)
                results[ticker] = {"price": None, "market_type": "unknown", "timestamp": now}
            else:
                price, mtype = result
                results[ticker] = {
                    "price": price,
                    "market_type": mtype,
                    "timestamp": now,
                }
                logger.debug("Snapshot {} : ${}", ticker, price)

        return results


@lru_cache(maxsize=1)
def get_market_fetcher() -> MarketDataFetcher:
    """Singleton du fetcher (important car ccxt maintient une session HTTP)."""
    return MarketDataFetcher()
