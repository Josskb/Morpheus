"""
tests/integration/test_pipeline.py
────────────────────────────────────
Tests d'intégration des trois premiers modules :
  - Module 1 : Collector (tweets → DB)
  - Module 2 : Market (snapshots prix)
  - Module 3 : NLP (analyse tweets → enrichissement DB)

Tous les appels externes sont mockés (Twitter, yfinance, ccxt).
La DB est une SQLite temporaire créée par la fixture `test_db`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from src.alerts.recap_service import RecapService
from src.collector.models import (
    AccountConfig,
    AccountsFileConfig,
    FiltersConfig,
    PollingConfig,
    RawTweet,
)
from src.collector.service import CollectorService
from src.collector.twitter_client import MockTwitterClient
from src.core.database import Account, MarketSnapshot, Prediction, Tweet
from src.market.snapshot_scheduler import SnapshotScheduler
from src.ml.labeler import OutcomeLabeler
from src.ml.service import MLScoringService
from src.nlp.processor import NLPProcessorService

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(usernames: list[str] = None) -> AccountsFileConfig:
    """Config minimale pour les tests."""
    usernames = usernames or ["test_trader"]
    return AccountsFileConfig(
        accounts=[
            AccountConfig(username=u, priority="high", enabled=True)
            for u in usernames
        ],
        polling=PollingConfig(interval_seconds=300, tweets_per_account=5),
        filters=FiltersConfig(
            exclude_retweets=True,
            exclude_replies=False,
            min_length=20,
            languages=["en"],
        ),
    )


def make_raw_tweet(
    text: str = "$BTC looking very bullish here, strong support at 40k 🚀",
    username: str = "test_trader",
    tweet_id: str = "111222333",
    is_retweet: bool = False,
) -> RawTweet:
    return RawTweet(
        tweet_id=tweet_id,
        username=username,
        text=text,
        tweeted_at=datetime.now(tz=timezone.utc),
        lang="en",
        like_count=100,
        retweet_count=20,
        reply_count=5,
        is_retweet=is_retweet,
    )


# ── Module 1 : Collector ──────────────────────────────────────────────────────

class TestCollectorModule:
    """Le collecteur synchronise les comptes et sauvegarde les tweets en DB."""

    @pytest.mark.asyncio
    async def test_sync_accounts_creates_db_records(self, test_db):
        """sync_accounts_to_db() insère les comptes YAML en DB."""
        config = make_config(["trader_a", "trader_b"])
        svc = CollectorService(client=MockTwitterClient(seed=0), config=config)

        await svc.sync_accounts_to_db()

        async with test_db() as session:
            result = await session.execute(select(Account))
            accounts = result.scalars().all()

        assert len(accounts) == 2
        usernames = {a.username for a in accounts}
        assert usernames == {"trader_a", "trader_b"}

    @pytest.mark.asyncio
    async def test_sync_accounts_idempotent(self, test_db):
        """sync_accounts_to_db() peut être appelé deux fois sans créer de doublons."""
        config = make_config(["trader_x"])
        svc = CollectorService(client=MockTwitterClient(seed=0), config=config)

        await svc.sync_accounts_to_db()
        await svc.sync_accounts_to_db()

        async with test_db() as session:
            result = await session.execute(select(Account))
            accounts = result.scalars().all()

        assert len(accounts) == 1

    @pytest.mark.asyncio
    async def test_poll_once_inserts_tweets(self, test_db):
        """poll_once() insère les tweets du client en DB."""
        config = make_config(["test_trader"])
        mock_client = MockTwitterClient(seed=42)
        svc = CollectorService(client=mock_client, config=config)

        await svc.sync_accounts_to_db()
        results = await svc.poll_once()

        total = sum(results.values())
        assert total > 0, "Au moins un tweet devrait être inséré"

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweets = result.scalars().all()

        assert len(tweets) == total
        assert all(t.account_id is not None for t in tweets)
        assert all(t.text for t in tweets)

    @pytest.mark.asyncio
    async def test_poll_deduplicates_tweets(self, test_db):
        """Un tweet avec le même tweet_id n'est inséré qu'une seule fois."""
        config = make_config(["test_trader"])
        fixed_tweet = make_raw_tweet(tweet_id="999888777")

        mock_client = AsyncMock(spec=MockTwitterClient)
        mock_client.get_recent_tweets.return_value = [fixed_tweet]

        svc = CollectorService(client=mock_client, config=config)
        await svc.sync_accounts_to_db()

        # Premier poll
        await svc.poll_once()
        # Deuxième poll avec le même tweet
        await svc.poll_once()

        async with test_db() as session:
            result = await session.execute(
                select(Tweet).where(Tweet.tweet_id == "999888777")
            )
            tweets = result.scalars().all()

        assert len(tweets) == 1, "Le tweet ne doit être inséré qu'une fois"

    @pytest.mark.asyncio
    async def test_poll_filters_retweets(self, test_db):
        """Les retweets sont filtrés et ne sont pas stockés en DB."""
        config = make_config(["test_trader"])
        retweet = make_raw_tweet(is_retweet=True, tweet_id="RT001")
        normal = make_raw_tweet(is_retweet=False, tweet_id="TW001")

        mock_client = AsyncMock(spec=MockTwitterClient)
        mock_client.get_recent_tweets.return_value = [retweet, normal]

        svc = CollectorService(client=mock_client, config=config)
        await svc.sync_accounts_to_db()
        await svc.poll_once()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweets = result.scalars().all()

        assert len(tweets) == 1
        assert tweets[0].tweet_id == "TW001"

    @pytest.mark.asyncio
    async def test_tweets_start_unprocessed(self, test_db):
        """Les tweets collectés ont nlp_processed=False par défaut."""
        config = make_config(["test_trader"])
        svc = CollectorService(client=MockTwitterClient(seed=1), config=config)

        await svc.sync_accounts_to_db()
        await svc.poll_once()

        async with test_db() as session:
            result = await session.execute(
                select(Tweet).where(Tweet.nlp_processed == False)  # noqa: E712
            )
            unprocessed = result.scalars().all()

        assert len(unprocessed) > 0, "Les tweets doivent commencer non-traités"


# ── Module 3 : NLP Processor ──────────────────────────────────────────────────

class TestNLPModule:
    """Le processeur NLP enrichit les tweets non traités en DB."""

    async def _insert_tweet(self, session_factory, text: str, tweet_id: str = "T001") -> int:
        """Insère un tweet et son compte directement en DB pour les tests."""
        async with session_factory() as session:
            account = Account(
                username="test_nlp_trader",
                display_name="NLP Test Trader",
                markets='["crypto"]',
                tags='[]',
                priority="high",
                enabled=True,
            )
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id=tweet_id,
                account_id=account.id,
                text=text,
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                nlp_processed=False,
            )
            session.add(tweet)
            await session.commit()
            return tweet.id

    @pytest.mark.asyncio
    async def test_process_pending_marks_tweets_processed(self, test_db):
        """process_pending() marque les tweets comme nlp_processed=True."""
        await self._insert_tweet(test_db, "$BTC bullish setup 🚀 accumulating here")

        nlp = NLPProcessorService(scheduler=None)
        count = await nlp.process_pending()

        assert count == 1

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.nlp_processed is True

    @pytest.mark.asyncio
    async def test_nlp_extracts_tickers(self, test_db):
        """Les tickers sont extraits et stockés en JSON dans la colonne tickers."""
        await self._insert_tweet(test_db, "$BTC and $ETH looking very bullish today!")

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        tickers = json.loads(tweet.tickers)
        assert "BTC" in tickers
        assert "ETH" in tickers

    @pytest.mark.asyncio
    async def test_nlp_classifies_bullish_tweet(self, test_db):
        """Un tweet bullish est classifié comme 'long'."""
        await self._insert_tweet(
            test_db,
            "Loading up $BTC here, very bullish setup. Strong support, accumulating 🚀",
        )

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.call_type == "long"
        assert tweet.sentiment > 0
        assert tweet.confidence is not None and tweet.confidence > 0

    @pytest.mark.asyncio
    async def test_nlp_classifies_bearish_tweet(self, test_db):
        """Un tweet bearish est classifié comme 'short'."""
        await self._insert_tweet(
            test_db,
            "Shorting $ETH here, bearish breakdown confirmed. Resistance holding 📉",
        )

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.call_type == "short"
        assert tweet.sentiment < 0

    @pytest.mark.asyncio
    async def test_nlp_extracts_target_price(self, test_db):
        """Le prix cible est extrait quand mentionné."""
        await self._insert_tweet(
            test_db,
            "$BTC looking great, target $50,000. Very bullish accumulating here 🚀",
        )

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.target_price == 50_000.0

    @pytest.mark.asyncio
    async def test_nlp_extracts_stop_loss(self, test_db):
        """Le stop loss est extrait quand mentionné."""
        await self._insert_tweet(
            test_db,
            "Long $BTC, SL $38,000. Bullish breakout expected 🚀",
        )

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.stop_loss == 38_000.0

    @pytest.mark.asyncio
    async def test_nlp_handles_tweet_without_ticker(self, test_db):
        """Un tweet sans ticker est traité sans erreur, confidence réduite."""
        await self._insert_tweet(
            test_db,
            "The market is looking very bullish today, great setup for longs!",
        )

        nlp = NLPProcessorService(scheduler=None)
        count = await nlp.process_pending()

        assert count == 1

        async with test_db() as session:
            result = await session.execute(select(Tweet))
            tweet = result.scalar_one()

        assert tweet.nlp_processed is True
        assert tweet.tickers is None  # pas de ticker trouvé
        assert tweet.confidence is not None and tweet.confidence > 0

    @pytest.mark.asyncio
    async def test_nlp_processes_multiple_tweets(self, test_db):
        """process_pending() traite tous les tweets non-traités en batch."""
        async with test_db() as session:
            account = Account(
                username="batch_trader",
                markets='[]',
                tags='[]',
                priority="medium",
                enabled=True,
            )
            session.add(account)
            await session.flush()

            for i in range(5):
                session.add(Tweet(
                    tweet_id=f"BATCH{i:03d}",
                    account_id=account.id,
                    text=f"$BTC tweet number {i}, bullish setup buying more 🚀",
                    lang="en",
                    tweeted_at=datetime.now(tz=timezone.utc),
                    nlp_processed=False,
                ))
            await session.commit()

        nlp = NLPProcessorService(scheduler=None)
        count = await nlp.process_pending()

        assert count == 5

        async with test_db() as session:
            result = await session.execute(
                select(Tweet).where(Tweet.nlp_processed == True)  # noqa: E712
            )
            processed = result.scalars().all()

        assert len(processed) == 5

    @pytest.mark.asyncio
    async def test_nlp_skips_already_processed(self, test_db):
        """process_pending() ignore les tweets déjà traités."""
        async with test_db() as session:
            account = Account(
                username="skip_trader",
                markets='[]',
                tags='[]',
                priority="medium",
                enabled=True,
            )
            session.add(account)
            await session.flush()

            session.add(Tweet(
                tweet_id="DONE001",
                account_id=account.id,
                text="$BTC bullish setup, buying and accumulating more 🚀",
                lang="en",
                tweeted_at=datetime.now(tz=timezone.utc),
                nlp_processed=True,   # déjà traité
            ))
            await session.commit()

        nlp = NLPProcessorService(scheduler=None)
        count = await nlp.process_pending()

        assert count == 0, "Aucun tweet ne devrait être re-traité"

    @pytest.mark.asyncio
    async def test_nlp_triggers_snapshot_scheduler(self, test_db):
        """Quand des tickers sont trouvés, le scheduler est appelé."""
        await self._insert_tweet(
            test_db,
            "$BTC and $ETH setup very bullish today, accumulating 🚀",
        )

        mock_scheduler = AsyncMock(spec=SnapshotScheduler)
        nlp = NLPProcessorService(scheduler=mock_scheduler)
        await nlp.process_pending()

        mock_scheduler.schedule_for_tweet.assert_called_once()
        call_kwargs = mock_scheduler.schedule_for_tweet.call_args
        scheduled_tickers = call_kwargs.kwargs.get("tickers") or call_kwargs.args[1]
        assert "BTC" in scheduled_tickers or "ETH" in scheduled_tickers

    @pytest.mark.asyncio
    async def test_nlp_no_scheduler_call_without_tickers(self, test_db):
        """Sans ticker détecté, le scheduler n'est pas appelé."""
        await self._insert_tweet(
            test_db,
            "The market is looking very bullish today with strong upward momentum!",
        )

        mock_scheduler = AsyncMock(spec=SnapshotScheduler)
        nlp = NLPProcessorService(scheduler=mock_scheduler)
        await nlp.process_pending()

        mock_scheduler.schedule_for_tweet.assert_not_called()

    @pytest.mark.asyncio
    async def test_nlp_respects_pre_filled_hints_from_source(self, test_db):
        """
        Un tweet dont tickers/sentiment/confidence sont déjà remplis par la
        source (StockTwits) ne doit pas se faire écraser ces champs par une
        ré-estimation lexicale — mais call_type/target_price/etc. sont quand
        même calculés normalement.
        """
        async with test_db() as session:
            account = Account(
                username="BTC.X", markets='["crypto"]', tags='[]',
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id="ST001",
                account_id=account.id,
                text="(@trader_x) loading up here, looks strong",
                tickers=json.dumps(["BTC.X"]),
                sentiment=0.6,
                confidence=0.6,
                tweeted_at=datetime.now(tz=timezone.utc),
                nlp_processed=False,
            )
            session.add(tweet)
            await session.commit()

        mock_scheduler = AsyncMock(spec=SnapshotScheduler)
        nlp = NLPProcessorService(scheduler=mock_scheduler)
        count = await nlp.process_pending()
        assert count == 1

        async with test_db() as session:
            result = await session.execute(select(Tweet).where(Tweet.tweet_id == "ST001"))
            tweet = result.scalar_one()

        # Hints respectés, pas recalculés lexicalement.
        assert json.loads(tweet.tickers) == ["BTC.X"]
        assert tweet.sentiment == pytest.approx(0.6)
        assert tweet.confidence == pytest.approx(0.6)
        # Le reste du pipeline NLP tourne quand même normalement.
        assert tweet.call_type is not None
        assert tweet.nlp_processed is True
        mock_scheduler.schedule_for_tweet.assert_called_once()

    @pytest.mark.asyncio
    async def test_nlp_fills_hints_when_source_did_not_provide_them(self, test_db):
        """Sans hints de la source (Mock/Twitter classique), le comportement lexical habituel s'applique."""
        await self._insert_tweet(
            test_db, "$BTC bullish setup, accumulating more 🚀", tweet_id="TW001",
        )

        nlp = NLPProcessorService(scheduler=None)
        await nlp.process_pending()

        async with test_db() as session:
            result = await session.execute(select(Tweet).where(Tweet.tweet_id == "TW001"))
            tweet = result.scalar_one()

        assert "BTC" in json.loads(tweet.tickers)
        assert tweet.sentiment is not None


# ── Module 2 : Market Fetcher ─────────────────────────────────────────────────

class TestMarketModule:
    """Le fetcher détecte le type de marché et normalise les tickers."""

    def test_detect_market_type_crypto(self):
        from src.market.fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("BTC") == "crypto"
        assert fetcher.detect_market_type("ETH") == "crypto"
        assert fetcher.detect_market_type("SOL") == "crypto"
        assert fetcher.detect_market_type("$BTC") == "crypto"
        assert fetcher.detect_market_type("BTC/USDT") == "crypto"

    def test_detect_market_type_stock(self):
        from src.market.fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        assert fetcher.detect_market_type("AAPL") == "stock"
        assert fetcher.detect_market_type("NVDA") == "stock"
        assert fetcher.detect_market_type("SPY") == "stock"

    def test_crypto_pair_normalization(self):
        from src.market.fetcher import CryptoFetcher
        f = CryptoFetcher.__new__(CryptoFetcher)
        assert f._normalize_pair("BTC") == "BTC/USDT"
        assert f._normalize_pair("eth") == "ETH/USDT"
        assert f._normalize_pair("$SOL") == "SOL/USDT"
        assert f._normalize_pair("BTC/USDT") == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_snapshot_returns_price_dict(self):
        """snapshot_tweet_tickers retourne bien {ticker: {price, market_type}}."""
        from src.market.fetcher import MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher._stock = MagicMock()
        fetcher._stock.get_price = AsyncMock(return_value=175.50)
        fetcher._crypto = MagicMock()
        fetcher._crypto.get_price = AsyncMock(return_value=42000.0)

        result = await fetcher.snapshot_tweet_tickers(
            ["AAPL", "BTC"],
            market_types={"AAPL": "stock", "BTC": "crypto"},
        )

        assert "AAPL" in result and "BTC" in result
        assert result["AAPL"]["price"] == 175.50
        assert result["BTC"]["price"] == 42000.0
        assert result["AAPL"]["market_type"] == "stock"
        assert result["BTC"]["market_type"] == "crypto"

    @pytest.mark.asyncio
    async def test_snapshot_handles_fetch_error(self):
        """snapshot_tweet_tickers retourne price=None si le fetch échoue."""
        from src.market.fetcher import MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher._stock = MagicMock()
        fetcher._stock.get_price = AsyncMock(side_effect=Exception("network error"))
        fetcher._crypto = MagicMock()
        fetcher._crypto.get_price = AsyncMock(return_value=42000.0)

        result = await fetcher.snapshot_tweet_tickers(
            ["AAPL", "BTC"],
            market_types={"AAPL": "stock", "BTC": "crypto"},
        )

        assert result["AAPL"]["price"] is None
        assert result["BTC"]["price"] == 42000.0


# ── Pipeline complet ──────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    Pipeline end-to-end :
    Collect (MockTwitter) → DB → NLP processor → tweets enrichis → scheduler appelé.
    """

    @pytest.mark.asyncio
    async def test_collect_then_nlp_enriches_all_tweets(self, test_db):
        """
        Après collect + NLP, tous les tweets en DB sont enrichis
        (nlp_processed=True, call_type non-null).
        """
        config = make_config(["pipeline_trader"])
        svc = CollectorService(client=MockTwitterClient(seed=7), config=config)

        await svc.sync_accounts_to_db()
        results = await svc.poll_once()
        total_collected = sum(results.values())
        assert total_collected > 0

        # Vérifie que tous les tweets sont non-traités après collect
        async with test_db() as session:
            res = await session.execute(
                select(Tweet).where(Tweet.nlp_processed == False)  # noqa: E712
            )
            unprocessed_before = res.scalars().all()
        assert len(unprocessed_before) == total_collected

        # Lance le NLP processor
        mock_scheduler = AsyncMock(spec=SnapshotScheduler)
        nlp = NLPProcessorService(scheduler=mock_scheduler)
        count = await nlp.process_pending()

        assert count == total_collected

        # Vérifie que tous les tweets sont maintenant traités
        async with test_db() as session:
            res = await session.execute(select(Tweet))
            all_tweets = res.scalars().all()

        assert all(t.nlp_processed is True for t in all_tweets)
        assert all(t.call_type is not None for t in all_tweets)
        assert all(t.sentiment is not None for t in all_tweets)
        assert all(t.confidence is not None for t in all_tweets)

    @pytest.mark.asyncio
    async def test_tweets_with_tickers_schedule_snapshots(self, test_db):
        """
        Les tweets qui contiennent des tickers déclenchent le scheduling de snapshots.
        """
        # Tweet avec ticker explicite ($BTC)
        config = make_config(["ticker_trader"])
        fixed_tweet = make_raw_tweet(
            text="$BTC breakout confirmed! Loading up here. Very bullish! 🚀🚀",
            username="ticker_trader",
            tweet_id="TICKER001",
        )

        mock_client = AsyncMock(spec=MockTwitterClient)
        mock_client.get_recent_tweets.return_value = [fixed_tweet]

        svc = CollectorService(client=mock_client, config=config)
        await svc.sync_accounts_to_db()
        await svc.poll_once()

        mock_scheduler = AsyncMock(spec=SnapshotScheduler)
        nlp = NLPProcessorService(scheduler=mock_scheduler)
        await nlp.process_pending()

        # Le scheduler doit avoir été appelé pour BTC
        mock_scheduler.schedule_for_tweet.assert_called_once()
        call_args = mock_scheduler.schedule_for_tweet.call_args
        tickers = call_args.kwargs.get("tickers") or call_args.args[1]
        assert "BTC" in tickers

    @pytest.mark.asyncio
    async def test_real_scheduler_does_not_raise_on_db_roundtrip_tweet(self, test_db):
        """
        Régression : un tweet relu depuis la DB doit rester comparable à un
        datetime.now(tz=timezone.utc) fraîchement créé dans
        SnapshotScheduler.schedule_for_tweet (pas de scheduler mocké ici,
        contrairement aux autres tests de ce fichier — c'est justement ce qui
        laissait passer le bug offset-naive/offset-aware).
        """
        config = make_config(["real_scheduler_trader"])
        fixed_tweet = make_raw_tweet(
            text="$BTC breakout confirmed, very bullish, loading up here 🚀",
            username="real_scheduler_trader", tweet_id="REALSCHED001",
        )
        mock_client = AsyncMock(spec=MockTwitterClient)
        mock_client.get_recent_tweets.return_value = [fixed_tweet]

        svc = CollectorService(client=mock_client, config=config)
        await svc.sync_accounts_to_db()
        await svc.poll_once()

        nlp = NLPProcessorService(scheduler=SnapshotScheduler())
        count = await nlp.process_pending()

        assert count == 1
        async with test_db() as session:
            tweet = (await session.execute(select(Tweet))).scalar_one()
        assert tweet.nlp_processed is True
        assert tweet.call_type == "long"

    @pytest.mark.asyncio
    async def test_multiple_accounts_all_processed(self, test_db):
        """
        Deux comptes → tweets des deux → tous traités par NLP.
        """
        config = make_config(["account_alpha", "account_beta"])
        mock_client = MockTwitterClient(seed=99)
        svc = CollectorService(client=mock_client, config=config)

        await svc.sync_accounts_to_db()
        results = await svc.poll_once()

        # Les deux comptes ont des tweets
        assert len(results) == 2
        total = sum(results.values())
        assert total > 0

        nlp = NLPProcessorService(scheduler=None)
        count = await nlp.process_pending()

        assert count == total

        async with test_db() as session:
            res = await session.execute(
                select(Tweet).where(Tweet.nlp_processed == True)  # noqa: E712
            )
            processed = res.scalars().all()

        assert len(processed) == total


# ── Module 4 : ML Scoring ──────────────────────────────────────────────────────

class TestMLModule:
    """
    Chaîne complète : tweet enrichi NLP → snapshot marché → scoring ML
    (repli heuristique, pas de modèle entraîné) → labeling → mise à jour des
    stats de fiabilité du compte.
    """

    async def _insert_scored_call(self, session_factory, tweet_id: str = "ML001") -> None:
        async with session_factory() as session:
            account = Account(
                username=f"ml_pipeline_trader_{tweet_id}", markets='["crypto"]', tags="[]",
                priority="high", enabled=True,
            )
            session.add(account)
            await session.flush()

            tweet = Tweet(
                tweet_id=tweet_id, account_id=account.id,
                text="$BTC breakout confirmed, very bullish, loading up here 🚀",
                tickers='["BTC"]', call_type="long",
                sentiment=0.7, confidence=0.8, urgency_score=0.4,
                tweeted_at=datetime.now(tz=timezone.utc), nlp_processed=True,
            )
            session.add(tweet)
            await session.flush()

            session.add(MarketSnapshot(
                tweet_id=tweet.id, ticker="BTC", market_type="crypto",
                price_at_tweet=40_000.0,
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_full_ml_chain_scores_labels_and_updates_account(self, test_db):
        await self._insert_scored_call(test_db)

        # Scoring : pas de modèle entraîné → repli heuristique.
        scored = await MLScoringService().score_pending()
        assert scored == 1

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet))).scalar_one()
            prediction = (await session.execute(select(Prediction))).scalar_one()
        assert tweet.ml_score is not None
        assert prediction.actual_profitable is None  # pas encore résolu

        # Le marché bouge, la fenêtre 24h se remplit (simulé, sans passer par
        # le vrai SnapshotScheduler qui est déjà testé ailleurs).
        async with test_db() as session:
            snap = (await session.execute(select(MarketSnapshot))).scalar_one()
            snap.change_24h = 12.5
            await session.commit()

        labeled = await OutcomeLabeler().label_pending()
        assert labeled == 1

        async with test_db() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
            account = (await session.execute(select(Account))).scalar_one()

        assert prediction.actual_profitable is True
        assert account.total_calls == 1
        assert account.win_rate == 1.0
        assert account.reliability_score == pytest.approx(4 / 7)

    @pytest.mark.asyncio
    async def test_ml_module_does_not_touch_alerts(self, test_db):
        """Module 4 reste découplé du Module 5 : aucun impact sur alerted_at."""
        await self._insert_scored_call(test_db, tweet_id="ML002")

        await MLScoringService().score_pending()

        async with test_db() as session:
            snap = (await session.execute(select(MarketSnapshot))).scalar_one()
            snap.change_24h = -5.0
            await session.commit()

        await OutcomeLabeler().label_pending()

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet))).scalar_one()
        assert tweet.alerted_at is None


# ── Module Récap Telegram ──────────────────────────────────────────────────────

class TestRecapModule:
    """Récap groupé : signaux forts par compte/ticker + digest des signaux faibles."""

    async def _insert_signal(self, session_factory, tweet_id: str, username: str, sentiment: float) -> None:
        async with session_factory() as session:
            account = Account(
                username=username, markets='["crypto"]', tags="[]",
                priority="medium", enabled=True,
            )
            session.add(account)
            await session.flush()

            session.add(Tweet(
                tweet_id=tweet_id, account_id=account.id, text=f"signal {username}",
                sentiment=sentiment, nlp_processed=True, tickers='["BTC"]',
                tweeted_at=datetime.now(tz=timezone.utc),
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_strong_and_weak_signals_go_to_different_recaps(self, test_db):
        """Un signal fort part dans le récap par compte, un faible dans le digest — pas les deux."""
        await self._insert_signal(test_db, "R1", "strong_trader", sentiment=0.9)
        await self._insert_signal(test_db, "R2", "weak_trader", sentiment=0.1)

        mock_bot = AsyncMock()
        recap = RecapService(bot=mock_bot, check_interval_seconds=60)
        recap._last_alert_recap_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        recap._last_digest_at = datetime.now(tz=timezone.utc) - timedelta(minutes=45)

        await recap._send_alert_recap_if_due()
        await recap._send_digest_if_due()

        strong_rows = mock_bot.send_recap_by_account.call_args[0][0]
        weak_rows = mock_bot.send_digest.call_args[0][0]

        assert {r[0] for r in strong_rows} == {"strong_trader"}
        assert {r[0] for r in weak_rows} == {"weak_trader"}

    @pytest.mark.asyncio
    async def test_recap_does_not_touch_alerted_at(self, test_db):
        """Module récap reste découplé d'AlertService : n'écrit jamais alerted_at."""
        await self._insert_signal(test_db, "R3", "strong_trader", sentiment=0.9)

        mock_bot = AsyncMock()
        recap = RecapService(bot=mock_bot, check_interval_seconds=60)
        recap._last_alert_recap_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        await recap._send_alert_recap_if_due()

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet).where(Tweet.tweet_id == "R3"))).scalar_one()
        assert tweet.alerted_at is None
