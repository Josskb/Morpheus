"""
market/snapshot_scheduler.py
──────────────────────────────
Planifie et exécute les snapshots de prix post-tweet.

Après chaque tweet détecté avec des tickers :
  - t+1h  → snapshot prix
  - t+4h  → snapshot prix
  - t+24h → snapshot prix
  - t+7d  → snapshot prix

Calcule ensuite les variations (%) et les stocke dans market_snapshots.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from ..core.database import AsyncSessionLocal, MarketSnapshot, Tweet
from .fetcher import MarketType, get_market_fetcher

# Fenêtres de tracking en minutes
WINDOWS: list[tuple[str, int]] = [
    ("1h", 60),
    ("4h", 240),
    ("24h", 1440),
    ("7d", 10080),
]


class SnapshotScheduler:
    """
    Maintient une queue de snapshots à effectuer et les exécute au bon moment.

    Simple et robuste : au démarrage, recharge les snapshots manquants
    (tweets passés dont les fenêtres ne sont pas encore remplies).
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._fetcher = get_market_fetcher()

    async def schedule_for_tweet(
        self,
        tweet_db_id: int,
        tickers: list[str],
        tweeted_at: datetime,
        market_types: dict[str, str] | None = None,
    ) -> None:
        """
        Ajoute tous les snapshots futurs pour un tweet dans la queue.
        Appelé juste après l'insertion d'un nouveau tweet.
        """
        now = datetime.now(tz=timezone.utc)

        for label, minutes in WINDOWS:
            target_time = tweeted_at + timedelta(minutes=minutes)
            if target_time <= now:
                # Fenêtre déjà passée → snapshot immédiat
                delay_seconds = 0
            else:
                delay_seconds = (target_time - now).total_seconds()

            # (priority = timestamp unix, payload = task info)
            priority = target_time.timestamp()
            await self._queue.put((
                priority,
                {
                    "tweet_db_id": tweet_db_id,
                    "tickers": tickers,
                    "market_types": market_types or {},
                    "window_label": label,
                    "target_time": target_time,
                }
            ))
            logger.debug(
                "Snapshot schedulé tweet={} {} dans {:.0f}s",
                tweet_db_id, label, delay_seconds,
            )

    async def _take_snapshot(self, task: dict) -> None:
        """Effectue le snapshot et met à jour la DB."""
        tweet_id = task["tweet_db_id"]
        tickers = task["tickers"]
        window = task["window_label"]
        market_types: dict[str, MarketType] = {
            k: v for k, v in task.get("market_types", {}).items()
            if v in ("stock", "crypto")
        }

        prices = await self._fetcher.snapshot_tweet_tickers(tickers, market_types or None)

        async with AsyncSessionLocal() as session:
            for ticker, data in prices.items():
                price_now = data.get("price")
                if price_now is None:
                    continue

                # Cherche ou crée le MarketSnapshot pour ce tweet+ticker
                result = await session.execute(
                    select(MarketSnapshot).where(
                        MarketSnapshot.tweet_id == tweet_id,
                        MarketSnapshot.ticker == ticker,
                    )
                )
                snap = result.scalar_one_or_none()

                if not snap:
                    # Cas rare (snapshot initial manquant) → crée
                    snap = MarketSnapshot(
                        tweet_id=tweet_id,
                        ticker=ticker,
                        market_type=data.get("market_type", "unknown"),
                    )
                    session.add(snap)
                    await session.flush()

                # Remplit la colonne correspondant à la fenêtre
                col_price = f"price_{window}"
                if hasattr(snap, col_price):
                    setattr(snap, col_price, price_now)

                # Calcule la variation si on a le prix de référence
                if snap.price_at_tweet and snap.price_at_tweet > 0:
                    change = (price_now - snap.price_at_tweet) / snap.price_at_tweet * 100
                    col_change = f"change_{window}"
                    if hasattr(snap, col_change):
                        setattr(snap, col_change, round(change, 4))
                    logger.info(
                        "Snapshot tweet={} {} {} : ${:.4f} ({:+.2f}%)",
                        tweet_id, ticker, window, price_now, change,
                    )

            await session.commit()

    async def run(self) -> None:
        """Boucle principale du scheduler."""
        self._running = True
        logger.info("SnapshotScheduler démarré.")

        while self._running:
            try:
                # Attente non-bloquante
                priority, task = await asyncio.wait_for(
                    self._queue.get(), timeout=60.0
                )
                target: datetime = task["target_time"]
                now = datetime.now(tz=timezone.utc)

                if target > now:
                    # Remet dans la queue et attend
                    await self._queue.put((priority, task))
                    await asyncio.sleep(min(30.0, (target - now).total_seconds()))
                    continue

                await self._take_snapshot(task)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue  # queue vide, on re-boucle
            except Exception as e:
                logger.exception("Erreur snapshot : {}", e)

    def stop(self) -> None:
        self._running = False

    async def reload_pending_from_db(self) -> int:
        """
        Au démarrage, recharge les snapshots manquants pour les tweets récents.
        Évite de perdre le tracking après un redémarrage.
        """
        import json

        count = 0
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=8)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tweet).where(
                    Tweet.tweeted_at >= cutoff,
                    Tweet.tickers.isnot(None),
                )
            )
            tweets = result.scalars().all()

            for tweet in tweets:
                tickers = json.loads(tweet.tickers or "[]")
                if not tickers:
                    continue
                await self.schedule_for_tweet(
                    tweet_db_id=tweet.id,
                    tickers=tickers,
                    tweeted_at=tweet.tweeted_at,
                )
                count += 1

        logger.info("{} tweets rechargés dans le scheduler.", count)
        return count
