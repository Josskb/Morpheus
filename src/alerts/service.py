"""
alerts/service.py
──────────────────
Service d'alertes : détecte les tweets actionnables et envoie les notifications
Telegram avec gestion du cooldown par compte.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from ..core.database import Account, AsyncSessionLocal, Tweet
from ..core.settings import get_settings
from .bot import TelegramBot
from .formatter import format_alert

_ACTIONABLE_CALLS = {"long", "short"}


class AlertService:
    """
    Boucle qui poll la DB toutes les N secondes et envoie les alertes
    pour les tweets qui passent le seuil de confiance.

    Conditions d'envoi :
      - nlp_processed = True
      - call_type IN ('long', 'short')
      - confidence >= MIN_CONFIDENCE_SCORE
      - alerted_at IS NULL (pas encore envoyé)
      - cooldown : aucune alerte envoyée pour ce compte dans les dernières
        ALERT_COOLDOWN_MINUTES minutes
    """

    def __init__(
        self,
        bot: TelegramBot | None = None,
        interval_seconds: int = 30,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._interval = interval_seconds
        self._running = False

    async def _is_in_cooldown(self, session, account_id: int) -> bool:
        """Vérifie si ce compte a déjà reçu une alerte dans la fenêtre de cooldown."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            minutes=self._settings.alert_cooldown_minutes
        )
        result = await session.execute(
            select(Tweet.id).where(
                Tweet.account_id == account_id,
                Tweet.alerted_at >= cutoff,
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def check_and_send(self) -> int:
        """
        Vérifie les tweets éligibles et envoie les alertes.
        Retourne le nombre d'alertes envoyées.
        """
        threshold = self._settings.min_confidence_score
        sent = 0

        async with AsyncSessionLocal() as session:
            # Tweets NLP traités, actionnables, non encore alertés
            result = await session.execute(
                select(Tweet, Account)
                .join(Account, Tweet.account_id == Account.id)
                .where(
                    Tweet.nlp_processed == True,        # noqa: E712
                    Tweet.call_type.in_(_ACTIONABLE_CALLS),
                    Tweet.confidence >= threshold,
                    Tweet.alerted_at.is_(None),
                )
                .order_by(
                    # Priorité : urgence décroissante, puis confiance décroissante
                    Tweet.urgency_score.desc().nulls_last(),
                    Tweet.confidence.desc(),
                )
                .limit(20)
            )
            candidates = result.all()

            for tweet, account in candidates:
                # Vérifie le cooldown pour ce compte
                if await self._is_in_cooldown(session, account.id):
                    logger.debug(
                        "Cooldown actif pour @{} — alerte ignorée (tweet {})",
                        account.username, tweet.tweet_id,
                    )
                    continue

                message = format_alert(tweet, account)
                ok = await self._bot.send(message)

                # Marque comme alerté même si Telegram est désactivé (mock log)
                tweet.alerted_at = datetime.now(tz=timezone.utc)
                # Flush immédiat pour que le prochain _is_in_cooldown le voie
                await session.flush()
                sent += 1

                if ok:
                    logger.info(
                        "Alerte envoyée : @{} {} {} conf={:.2f}",
                        account.username,
                        tweet.call_type,
                        json_tickers(tweet.tickers),
                        tweet.confidence,
                    )
                else:
                    logger.info(
                        "Alerte loggée (Telegram non configuré) : @{} {} {}",
                        account.username,
                        tweet.call_type,
                        json_tickers(tweet.tickers),
                    )

            await session.commit()

        return sent

    async def run(self) -> None:
        """Boucle principale du service d'alertes."""
        self._running = True
        await self._bot.start()
        logger.info(
            "AlertService démarré (seuil={}, cooldown={}min, intervalle={}s).",
            self._settings.min_confidence_score,
            self._settings.alert_cooldown_minutes,
            self._interval,
        )

        while self._running:
            try:
                count = await self.check_and_send()
                if count:
                    logger.info("{} alerte(s) envoyée(s).", count)
            except Exception as e:
                logger.exception("Erreur dans la boucle d'alertes : {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("AlertService arrêté.")


def json_tickers(tickers_json: str | None) -> str:
    """Extrait les tickers depuis le JSON pour les logs."""
    if not tickers_json:
        return "[]"
    try:
        import json
        return str(json.loads(tickers_json))
    except Exception:
        return tickers_json
