"""
Enhanced Alert Service with ML Scoring
Service d'alertes intégrant le scoring ML pour des recommandations meilleures
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from src.core.database import Account, AsyncSessionLocal, Tweet
from src.core.settings import get_settings
from .telegram_bot import TelegramBot
from .formatter_ml import format_alert_with_ml, get_ml_alert_score

_ACTIONABLE_CALLS = {"long", "short"}

# Thresholds
MIN_ML_SCORE = 0.65  # Minimum ML score for alert


class AlertServiceML:
    """
    Service d'alertes avec scoring ML.

    Envoie les alertes en fonction de:
    - Confiance NLP >= seuil
    - Score ML >= seuil
    - Pas en cooldown
    """

    def __init__(
        self,
        bot: TelegramBot | None = None,
        interval_seconds: int = 30,
        min_ml_score: float = MIN_ML_SCORE,
    ) -> None:
        self._settings = get_settings()
        self._bot = bot or TelegramBot()
        self._interval = interval_seconds
        self._running = False
        self._min_ml_score = min_ml_score

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
        Vérifie les tweets éligibles avec ML scoring et envoie les alertes.
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
                    Tweet.nlp_processed == True,  # noqa: E712
                    Tweet.call_type.in_(_ACTIONABLE_CALLS),
                    Tweet.confidence >= threshold,
                    Tweet.alerted_at.is_(None),
                )
                .order_by(
                    Tweet.urgency_score.desc().nulls_last(),
                    Tweet.confidence.desc(),
                )
                .limit(20)
            )
            candidates = result.all()

            for tweet, account in candidates:
                # Vérifie le cooldown
                if await self._is_in_cooldown(session, account.id):
                    logger.debug(
                        "Cooldown actif pour @{} — alerte ignorée (tweet {})",
                        account.username, tweet.tweet_id,
                    )
                    continue

                # **NEW**: ML Scoring
                ml_score = get_ml_alert_score(tweet, account)

                if ml_score < self._min_ml_score:
                    logger.debug(
                        "ML score trop bas pour @{}: {:.2f} < {:.2f}",
                        account.username, ml_score, self._min_ml_score,
                    )
                    continue

                # Format avec ML
                message = format_alert_with_ml(tweet, account)
                ok = await self._bot.send(message)

                # Mark as alerted
                tweet.alerted_at = datetime.now(tz=timezone.utc)
                await session.flush()
                sent += 1

                if ok:
                    logger.info(
                        "Alerte ML envoyée : @{} | ML Score: {:.2f} | Conf: {:.2f}",
                        account.username,
                        ml_score,
                        tweet.confidence,
                    )
                else:
                    logger.info(
                        "Alerte loggée (pas de Telegram) : @{} | ML: {:.2f}",
                        account.username,
                        ml_score,
                    )

            await session.commit()

        return sent

    async def run(self) -> None:
        """Boucle principale du service d'alertes ML."""
        self._running = True
        await self._bot.start()
        logger.info(
            "AlertServiceML démarré (seuil_nlp={}, seuil_ml={:.2f}, cooldown={}min)",
            self._settings.min_confidence_score,
            self._min_ml_score,
            self._settings.alert_cooldown_minutes,
        )

        while self._running:
            try:
                count = await self.check_and_send()
                if count:
                    logger.info("{} alerte(s) ML envoyée(s).", count)
            except Exception as e:
                logger.exception("Erreur dans la boucle d'alertes ML : {}", e)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        await self._bot.stop()
        logger.info("AlertServiceML arrêté.")


def json_tickers(tickers_json: str | None) -> str:
    """Extrait les tickers depuis le JSON pour les logs."""
    if not tickers_json:
        return "[]"
    try:
        import json
        return str(json.loads(tickers_json))
    except Exception:
        return tickers_json
