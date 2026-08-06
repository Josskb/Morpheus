"""
collector/trending_service.py
────────────────────────────────
Signal d'attention/momentum : surveille le classement ApeWisdom (mentions
Reddit + 4chan) et alerte sur les tickers dont les mentions explosent par
rapport à il y a 24h. Événement individuel notable, comme un rapport
short-seller — pas de récap groupé, pas de passage par le pipeline NLP/DB
(ce signal n'est pas un "call" ni un sentiment déclaré, juste une mesure
d'attention agrégée).

Persistance : data/trending_seen.json — {ticker: dernière alerte ISO} pour
le cooldown, évite de re-alerter sur le même ticker à chaque tick.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from ..alerts.bot import TelegramBot
from ..core.settings import DATA_DIR, get_settings
from .apewisdom_client import ApeWisdomClient

DEFAULT_STATE_FILE = DATA_DIR / "trending_seen.json"

_FILTERS: list[tuple[str, str]] = [
    ("all-crypto", "Crypto"),
    ("all-stocks", "Bourse"),
]


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
        notables. Retourne le nombre d'alertes envoyées.
        """
        state = self._load_state()
        sent = 0

        for filter_name, market_label in _FILTERS:
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
                state[ticker] = datetime.now(tz=timezone.utc).isoformat()
                sent += 1

        if sent:
            self._save_state(state)

        return sent

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
            except Exception as e:
                logger.exception("Erreur dans la boucle Trending : {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("TrendingService arrêté.")
