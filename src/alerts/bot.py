"""
alerts/bot.py
──────────────
Wrapper minimal autour de python-telegram-bot v21.
Gère l'initialisation/shutdown et les erreurs réseau.
"""

from __future__ import annotations

import html

from loguru import logger
from telegram import Bot
from telegram.error import TelegramError

from ..core.settings import get_settings


def _tradingview_url(symbol: str) -> str | None:
    """
    Construit un lien TradingView pour un ticker US/crypto.
    Retourne None pour les tickers internationaux (ex: "000660.KS", "ALKAL.PA")
    qui n'ont pas d'URL TradingView directement dérivable du symbole.
    """
    if symbol.endswith(".X"):
        return f"https://www.tradingview.com/symbols/{symbol[:-2]}USD/"
    if "." in symbol:
        return None
    return f"https://www.tradingview.com/symbols/{symbol}/"


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

    async def send_digest(self, rows: list[tuple[str, str, float]]) -> bool:
        """
        Envoie un récap groupé de tweets (généralement ceux sous le seuil
        d'alerte individuelle), groupés par compte.
        rows : liste de (username, text, sentiment).
        """
        if not rows:
            return False

        lines = ["🗞 <b>Récap Morpheus</b>"]
        current_user: str | None = None

        for username, text, sentiment in rows:
            if username != current_user:
                lines.append(f"\n👤 <b>@{html.escape(username)}</b>")
                current_user = username

            emoji = "🟢" if sentiment > 0.05 else "🔴" if sentiment < -0.05 else "⚪"
            snippet = " ".join(text.split())
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            lines.append(f"{emoji} {html.escape(snippet)}")

        lines.append("\n<i>Morpheus Finance Bot — récap</i>")
        return await self.send("\n".join(lines))

    async def send_recap_by_account(
        self, rows: list[tuple[str, str, float, float | None]], period_hours: int
    ) -> bool:
        """
        Récap des signaux forts groupés par compte.
        rows : liste de (username, text, sentiment, reliability_score),
               triée par username puis |sentiment| DESC.
        """
        if not rows:
            return False

        lines = [f"📢 <b>Signaux forts — {period_hours}h</b>"]
        current_user: str | None = None

        for username, text, sentiment, reliability_score in rows:
            if username != current_user:
                star = " ⭐" if reliability_score is not None and reliability_score >= 0.6 else ""
                lines.append(f"\n👤 <b>@{html.escape(username)}</b>{star}")
                current_user = username

            emoji = "🟢" if sentiment > 0 else "🔴"
            snippet = " ".join(text.split())
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            lines.append(f"{emoji} [{sentiment:+.2f}] {html.escape(snippet)}")

        lines.append("\n<i>Morpheus Finance Bot — récap comptes</i>")
        return await self.send("\n".join(lines))

    async def send_recap_by_ticker(
        self, rows: list[tuple[str, int, float]], period_hours: int
    ) -> bool:
        """
        Récap des signaux forts agrégés par titre, séparé en deux sections :
        Bourse (stocks) et Crypto (symboles .X).
        rows : liste de (ticker, count, avg_sentiment), triée par count DESC.
        """
        if not rows:
            return False

        crypto = [(t, c, s) for t, c, s in rows if t.endswith(".X")]
        stocks = [(t, c, s) for t, c, s in rows if not t.endswith(".X")]

        lines = [f"📊 <b>Signaux par titre — {period_hours}h</b>"]

        def _format_row(ticker: str, count: int, avg_sentiment: float) -> str:
            avg_sentiment = avg_sentiment or 0.0
            emoji = "🟢" if avg_sentiment > 0.05 else "🔴" if avg_sentiment < -0.05 else "⚪"
            url = _tradingview_url(ticker)
            label = html.escape(ticker)
            label_html = f'<a href="{url}"><b>{label}</b></a>' if url else f"<b>{label}</b>"
            return (
                f"{emoji} {label_html}"
                f" — {count} signal{'s' if count > 1 else ''},"
                f" sentiment moyen {avg_sentiment:+.2f}"
            )

        if stocks:
            lines.append("\n🏦 <b>Bourse</b>")
            for row in stocks:
                lines.append(_format_row(*row))

        if crypto:
            lines.append("\n₿ <b>Crypto</b>")
            for row in crypto:
                lines.append(_format_row(*row))

        lines.append("\n<i>Morpheus Finance Bot — récap titres</i>")
        return await self.send("\n".join(lines))

    async def send_daily_recap(self, rows: list[tuple[str, int, float]]) -> bool:
        """
        Envoie le récap quotidien des titres qui ont le plus fait parler
        d'eux sur les dernières 24h.
        rows : liste de (username, mentions, avg_sentiment), triée par
        nombre de mentions décroissant.
        """
        if not rows:
            return False

        lines = [
            "📊 <b>Récap quotidien Morpheus</b>",
            "Titres les plus mentionnés sur les dernières 24h :\n",
        ]

        for username, mentions, avg_sentiment in rows:
            avg_sentiment = avg_sentiment or 0.0
            emoji = "🟢" if avg_sentiment > 0.05 else "🔴" if avg_sentiment < -0.05 else "⚪"
            url = _tradingview_url(username)
            label = html.escape(username)
            label_html = f'<a href="{url}"><b>{label}</b></a>' if url else f"<b>{label}</b>"
            lines.append(
                f"{emoji} {label_html} — {mentions} mentions, "
                f"sentiment moyen {avg_sentiment:+.2f}"
            )

        lines.append("\n<i>Morpheus Finance Bot — récap quotidien</i>")
        return await self.send("\n".join(lines))
