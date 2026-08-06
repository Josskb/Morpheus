"""
collector/trending_service.py
────────────────────────────────
Signal d'attention/momentum : surveille le classement ApeWisdom (mentions
Reddit + 4chan) et alerte sur les tickers dont les mentions explosent par
rapport à il y a 24h. Événement individuel notable, comme un rapport
short-seller — pas de récap groupé, pas de passage par le pipeline
RawTweet/NLP (ce signal n'est pas un "call" ni un sentiment déclaré, juste
une mesure d'attention agrégée).

Chaque spike alerté capture aussi un prix de référence (TrendingSnapshot),
puis suivi à 1h/4h/24h/7d comme MarketSnapshot le fait pour les tweets —
pour savoir si un spike de mentions est effectivement suivi d'un mouvement
de prix, pas juste "ça spike".

Persistance : data/trending_seen.json — {ticker: dernière alerte ISO} pour
le cooldown, évite de re-alerter/re-snapshoter le même ticker à chaque tick.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy import or_, select

from ..alerts.bot import TelegramBot
from ..core.database import AsyncSessionLocal, TrendingSnapshot
from ..core.settings import DATA_DIR, get_settings
from ..market.fetcher import get_market_fetcher
from ..market.snapshot_scheduler import WINDOWS
from .apewisdom_client import ApeWisdomClient

DEFAULT_STATE_FILE = DATA_DIR / "trending_seen.json"

# (filtre ApeWisdom, libellé affiché dans l'alerte, market_type pour le fetcher)
_FILTERS: list[tuple[str, str, str]] = [
    ("all-crypto", "Crypto", "crypto"),
    ("all-stocks", "Bourse", "stock"),
]


def _fetcher_ticker(ticker: str, market_type: str) -> str:
    """ApeWisdom suffixe les tickers crypto en .X (BTC.X) ; le fetcher attend BTC."""
    return ticker.removesuffix(".X") if market_type == "crypto" else ticker


class TrendingService:
    def __init__(
        self,
        client: ApeWisdomClient | None = None,
        bot: TelegramBot | None = None,
        state_file: Path | None = None,
        interval_seconds: int = 600,
    ) -> None:
        self._settings = get_settings()
        self._client = client or ApeWisdomClient()
        self._bot = bot or TelegramBot()
        self._state_file = state_file or DEFAULT_STATE_FILE
        self._interval = interval_seconds
        self._running = False

    def _load_state(self) -> dict[str, str]:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {}

    def _save_state(self, state: dict[str, str]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(state, indent=2))

    def _is_notable(self, mentions: int, mentions_24h_ago: int) -> bool:
        if mentions < self._settings.trending_min_mentions:
            return False
        if mentions_24h_ago == 0:
            return True
        return mentions / mentions_24h_ago >= self._settings.trending_spike_multiplier

    def _cooldown_elapsed(self, state: dict[str, str], ticker: str) -> bool:
        last = state.get(ticker)
        if last is None:
            return True
        last_dt = datetime.fromisoformat(last)
        cooldown = timedelta(hours=self._settings.trending_alert_cooldown_hours)
        return datetime.now(tz=timezone.utc) - last_dt >= cooldown

    async def check_once(self) -> int:
        """
        Vérifie les classements crypto + bourse, alerte sur les tickers
        notables et capture un prix de référence pour chacun. Retourne le
        nombre d'alertes envoyées.
        """
        state = self._load_state()
        sent = 0

        async with AsyncSessionLocal() as session:
            for filter_name, market_label, market_type in _FILTERS:
                entries = await self._client.get_trending(filter_name)
                for entry in entries:
                    ticker = entry["ticker"]
                    if not self._is_notable(entry["mentions"], entry["mentions_24h_ago"]):
                        continue
                    if not self._cooldown_elapsed(state, ticker):
                        continue

                    await self._bot.send_trending_alert(
                        ticker=ticker,
                        name=entry["name"],
                        mentions=entry["mentions"],
                        mentions_24h_ago=entry["mentions_24h_ago"],
                        market=market_label,
                    )

                    price, _ = await get_market_fetcher().get_price(
                        _fetcher_ticker(ticker, market_type), market_type,
                    )
                    now = datetime.now(tz=timezone.utc)
                    session.add(TrendingSnapshot(
                        ticker=ticker,
                        market_type=market_type,
                        mentions_at_detection=entry["mentions"],
                        mentions_24h_ago_at_detection=entry["mentions_24h_ago"],
                        detected_at=now,
                        price_at_detection=price,
                    ))

                    state[ticker] = now.isoformat()
                    sent += 1

            await session.commit()

        if sent:
            self._save_state(state)

        return sent

    async def _update_pending_snapshots(self) -> None:
        """Remplit price_Xh/change_Xh pour les fenêtres échues et pas encore résolues."""
        now = datetime.now(tz=timezone.utc)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TrendingSnapshot).where(
                    or_(
                        TrendingSnapshot.change_1h.is_(None),
                        TrendingSnapshot.change_4h.is_(None),
                        TrendingSnapshot.change_24h.is_(None),
                        TrendingSnapshot.change_7d.is_(None),
                    )
                )
            )
            pending = result.scalars().all()

            for snap in pending:
                if snap.price_at_detection is None:
                    continue

                for label, minutes in WINDOWS:
                    change_col = f"change_{label}"
                    if getattr(snap, change_col) is not None:
                        continue
                    if snap.detected_at + timedelta(minutes=minutes) > now:
                        continue

                    price_now, _ = await get_market_fetcher().get_price(
                        _fetcher_ticker(snap.ticker, snap.market_type), snap.market_type,
                    )
                    if price_now is None:
                        continue

                    change = (price_now - snap.price_at_detection) / snap.price_at_detection * 100
                    setattr(snap, f"price_{label}", price_now)
                    setattr(snap, change_col, round(change, 4))

            await session.commit()

    async def generate_report(self) -> dict[str, dict]:
        """
        Stats agrégées par fenêtre résolue (n, win_rate, avg_change), et par
        marché. Une fenêtre sans aucune donnée résolue est absente du
        rapport plutôt que d'y figurer avec une division par zéro.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(TrendingSnapshot))
            snapshots = result.scalars().all()

        report: dict[str, dict] = {}
        for label, _ in WINDOWS:
            change_attr = f"change_{label}"
            resolved = [s for s in snapshots if getattr(s, change_attr) is not None]
            if not resolved:
                continue

            by_market: dict[str, list[float]] = {}
            for s in resolved:
                by_market.setdefault(s.market_type, []).append(getattr(s, change_attr))

            report[label] = {
                "n": len(resolved),
                "win_rate": sum(1 for s in resolved if getattr(s, change_attr) > 0) / len(resolved),
                "avg_change": sum(getattr(s, change_attr) for s in resolved) / len(resolved),
                "by_market": {
                    market: {
                        "n": len(changes),
                        "win_rate": sum(1 for c in changes if c > 0) / len(changes),
                        "avg_change": sum(changes) / len(changes),
                    }
                    for market, changes in by_market.items()
                },
            }

        return report

    async def run(self) -> None:
        self._running = True
        await self._bot.start()
        logger.info(
            "TrendingService démarré (min_mentions={}, spike x{}, cooldown={}h, intervalle={}s).",
            self._settings.trending_min_mentions,
            self._settings.trending_spike_multiplier,
            self._settings.trending_alert_cooldown_hours,
            self._interval,
        )

        while self._running:
            try:
                sent = await self.check_once()
                if sent:
                    logger.info("{} alerte(s) de tendance envoyée(s).", sent)
                await self._update_pending_snapshots()
            except Exception as e:
                logger.exception("Erreur dans la boucle Trending : {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("TrendingService arrêté.")
