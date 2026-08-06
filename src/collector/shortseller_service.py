"""
collector/shortseller_service.py
───────────────────────────────────
Service dédié aux rapports short-sellers : scrape les sites configurés,
envoie une alerte Telegram immédiate pour chaque nouveau rapport — hors
du système de récap périodique (un nouveau rapport est un événement rare
et à haute valeur, pas un signal à agréger).

Ne touche jamais radar.db : la persistance (URLs déjà vues) passe par
shortseller_client.py, un fichier JSON local, pas la DB SQL.
"""

from __future__ import annotations

import asyncio

import yaml
from loguru import logger

from ..alerts.bot import TelegramBot
from ..core.settings import get_settings
from .shortseller_client import ShortSellerClient


def _load_firms() -> list[dict]:
    path = get_settings().shortsellers_config
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("firms", [])


class ShortSellerService:
    def __init__(
        self,
        client: ShortSellerClient | None = None,
        bot: TelegramBot | None = None,
        interval_seconds: int = 300,
    ) -> None:
        self._client = client or ShortSellerClient()
        self._bot = bot or TelegramBot()
        self._firms = _load_firms()
        self._interval = interval_seconds
        self._running = False

    async def check_once(self) -> int:
        """
        Vérifie tous les firms configurés et envoie une alerte pour chaque
        nouveau rapport. Retourne le nombre d'alertes envoyées.
        """
        sent = 0
        for firm in self._firms:
            try:
                new_reports = await self._client.get_new_reports(
                    name=firm["name"],
                    report_url=firm["report_url"],
                    link_contains=firm["link_contains"],
                )
            except Exception as e:
                logger.error("Erreur ShortSeller {} : {}", firm.get("name"), e)
                continue

            for report in new_reports:
                await self._bot.send_shortseller_alert(
                    firm=firm["name"],
                    emoji=firm.get("emoji", "🐻"),
                    title=report["title"],
                    url=report["url"],
                )
                sent += 1

        return sent

    async def run(self) -> None:
        self._running = True
        await self._bot.start()
        logger.info(
            "ShortSellerService démarré ({} firms, intervalle={}s).",
            len(self._firms), self._interval,
        )

        while self._running:
            try:
                sent = await self.check_once()
                if sent:
                    logger.info("{} alerte(s) short-seller envoyée(s).", sent)
            except Exception as e:
                logger.exception("Erreur dans la boucle ShortSeller : {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("ShortSellerService arrêté.")
