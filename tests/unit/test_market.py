"""
tests/unit/test_market.py
──────────────────────────
Tests unitaires du module market.
Mocke les appels yfinance et ccxt pour être hors-ligne.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.market.fetcher import CryptoFetcher, MarketDataFetcher, StockFetcher


class TestMarketTypeDetection:
    def setup_method(self):
        self.fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        # Init sans vraie connexion
        MarketDataFetcher.CRYPTO_TICKERS  # access class attr

    def test_detects_btc_as_crypto(self):
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("BTC") == "crypto"
        assert fetcher.detect_market_type("$BTC") == "crypto"
        assert fetcher.detect_market_type("BTC/USDT") == "crypto"

    def test_detects_eth_as_crypto(self):
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("ETH") == "crypto"

    def test_detects_aapl_as_stock(self):
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("AAPL") == "stock"
        assert fetcher.detect_market_type("$AAPL") == "stock"

    def test_detects_spy_as_stock(self):
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("SPY") == "stock"

    def test_unknown_defaults_to_stock(self):
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("UNKNWN") == "stock"


class TestCryptoPairNormalization:
    def test_normalizes_bare_ticker(self):
        fetcher = CryptoFetcher.__new__(CryptoFetcher)
        assert fetcher._normalize_pair("BTC") == "BTC/USDT"
        assert fetcher._normalize_pair("eth") == "ETH/USDT"
        assert fetcher._normalize_pair("$SOL") == "SOL/USDT"

    def test_keeps_pair_as_is(self):
        fetcher = CryptoFetcher.__new__(CryptoFetcher)
        assert fetcher._normalize_pair("BTC/USDT") == "BTC/USDT"
        assert fetcher._normalize_pair("ETH/BTC") == "ETH/BTC"


class TestStockFetcher:
    @pytest.mark.asyncio
    async def test_returns_price_on_success(self):
        fetcher = StockFetcher()
        with patch.object(StockFetcher, "_fetch_price", return_value=175.50):
            price = await fetcher.get_price("AAPL")
        assert price == 175.50

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        # get_price attrape l'exception en interne et retourne None
        fetcher = StockFetcher()
        with patch.object(StockFetcher, "_fetch_price", side_effect=Exception("network error")):
            price = await fetcher.get_price("INVALID")
        assert price is None
