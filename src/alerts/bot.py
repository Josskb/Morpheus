"""
alerts/bot.py
──────────────
Wrapper minimal autour de python-telegram-bot v21.
Gère l'initialisation/shutdown et les erreurs réseau.
"""

from __future__ import annotations

from loguru import logger
from telegram import Bot
from telegram.error import TelegramError

from ..core.settings import get_settings


class TelegramBot:
    """Client Telegram réutilisable pour envoyer des messages."""

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._bot: Bot | None = None
        self._configured = bool(self._token and self._chat_id)

        if not self._configured:
            logger.warning(
                "TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant — "
                "les alertes seront loggées uniquement."
            )

    async def start(self) -> None:
        if self._configured:
            self._bot = Bot(token=self._token)
            await self._bot.initialize()
            logger.info("Bot Telegram initialisé.")

    async def stop(self) -> None:
        if self._bot:
            await self._bot.shutdown()
            self._bot = None

    async def send(self, text: str) -> bool:
        """
        Envoie un message HTML au chat configuré.
        Retourne True si envoyé, False si erreur ou non configuré.
        """
        if not self._configured or not self._bot:
            logger.info("[TELEGRAM MOCK]\n{}", text)
            return False

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return True
        except TelegramError as e:
            logger.error("Erreur Telegram : {}", e)
            return False
