"""
collector/service.py
─────────────────────
Service de collecte : orchestre le polling, filtre les tweets,
et persiste en base via SQLAlchemy.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import Account, AsyncSessionLocal, Tweet
from ..core.settings import get_settings
from .config_loader import get_enabled_usernames, load_accounts_config
from .models import AccountsFileConfig, RawTweet
from .twitter_client import TwitterClientBase, create_twitter_client


class CollectorService:
    """
    Boucle de polling principale.

    Lifecycle :
      1. setup()  → sync les comptes YAML → DB
      2. run()    → boucle infinie de polling
      3. stop()   → arrêt propre
    """

    def __init__(
        self,
        client: TwitterClientBase | None = None,
        config: AccountsFileConfig | None = None,
    ) -> None:
        self._client = client or create_twitter_client()
        self._config = config or load_accounts_config()
        self._running = False
        self._settings = get_settings()

    # ── Gestion des comptes ───────────────────────────────────────────────────

    async def sync_accounts_to_db(self) -> None:
        """
        Synchronise la liste des comptes YAML avec la table `accounts`.
        Insère les nouveaux, met à jour les existants.
        """
        async with AsyncSessionLocal() as session:
            for acc_cfg in self._config.accounts:
                result = await session.execute(
                    select(Account).where(Account.username == acc_cfg.username)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.display_name = acc_cfg.display_name
                    existing.markets = json.dumps(acc_cfg.markets)
                    existing.tags = json.dumps(acc_cfg.tags)
                    existing.priority = acc_cfg.priority
                    existing.enabled = acc_cfg.enabled
                    logger.debug("Compte mis à jour : @{}", acc_cfg.username)
                else:
                    session.add(Account(
                        username=acc_cfg.username,
                        display_name=acc_cfg.display_name,
                        markets=json.dumps(acc_cfg.markets),
                        tags=json.dumps(acc_cfg.tags),
                        priority=acc_cfg.priority,
                        enabled=acc_cfg.enabled,
                    ))
                    logger.info("Nouveau compte ajouté : @{}", acc_cfg.username)

            await session.commit()
        logger.info("Synchronisation comptes terminée.")

    # ── Filtrage ─────────────────────────────────────────────────────────────

    def _should_keep(self, tweet: RawTweet) -> bool:
        """Applique les filtres définis dans accounts.yaml."""
        f = self._config.filters

        if f.exclude_retweets and tweet.is_retweet:
            return False
        if len(tweet.text) < f.min_length:
            return False
        if f.languages and tweet.lang and tweet.lang not in f.languages:
            return False
        return True

    # ── Persistance ──────────────────────────────────────────────────────────

    async def _save_tweets(
        self, session: AsyncSession, tweets: list[RawTweet], account_id: int
    ) -> tuple[int, int]:
        """
        Sauvegarde les tweets en base (ignore les doublons via tweet_id unique).
        Retourne (inserted, skipped).
        """
        inserted = skipped = 0

        for raw in tweets:
            # Vérifie si le tweet existe déjà
            exists = await session.execute(
                select(Tweet.id).where(Tweet.tweet_id == raw.tweet_id)
            )
            if exists.scalar_one_or_none():
                skipped += 1
                continue

            session.add(Tweet(
                tweet_id=raw.tweet_id,
                account_id=account_id,
                text=raw.text,
                lang=raw.lang,
                like_count=raw.like_count,
                retweet_count=raw.retweet_count,
                reply_count=raw.reply_count,
                tweeted_at=raw.tweeted_at,
                is_retweet=raw.is_retweet,
                # Certaines sources (StockTwits) fournissent déjà ces champs —
                # le NLP les respectera au lieu de les recalculer (voir processor.py).
                tickers=json.dumps(raw.tickers) if raw.tickers else None,
                sentiment=raw.sentiment,
                confidence=raw.confidence,
            ))
            inserted += 1

        return inserted, skipped

    # ── Fetch d'un compte ─────────────────────────────────────────────────────

    async def fetch_account(
        self,
        username: str,
        session: AsyncSession,
        since: datetime | None = None,
    ) -> int:
        """
        Fetch et sauvegarde les tweets d'un compte.
        Retourne le nombre de nouveaux tweets insérés.
        """
        # Récupère l'account_id depuis la DB
        result = await session.execute(
            select(Account).where(Account.username == username)
        )
        account = result.scalar_one_or_none()
        if not account:
            logger.warning("Compte @{} non trouvé en DB, skip.", username)
            return 0

        polling = self._config.polling
        raw_tweets = await self._client.get_recent_tweets(
            username=username,
            max_results=polling.tweets_per_account,
            since=since,
        )

        filtered = [t for t in raw_tweets if self._should_keep(t)]
        dropped = len(raw_tweets) - len(filtered)
        if dropped:
            logger.debug("@{} : {} tweets filtrés", username, dropped)

        inserted, skipped = await self._save_tweets(session, filtered, account.id)
        if inserted:
            logger.info(
                "@{} : +{} nouveaux tweets ({} ignorés/doublons)",
                username, inserted, skipped + dropped,
            )
        return inserted

    # ── Boucle principale ─────────────────────────────────────────────────────

    async def poll_once(self) -> dict[str, int]:
        """
        Effectue un tour de polling sur tous les comptes actifs.
        Retourne un dict {username: tweets_inserted}.
        """
        usernames = get_enabled_usernames(self._config)
        if not usernames:
            logger.warning("Aucun compte actif configuré.")
            return {}

        lookback = timedelta(hours=self._config.polling.lookback_hours)
        since = datetime.now(tz=timezone.utc) - lookback

        results: dict[str, int] = {}
        async with AsyncSessionLocal() as session:
            for username in usernames:
                try:
                    count = await self.fetch_account(username, session, since=since)
                    results[username] = count
                except Exception as e:
                    logger.error("Erreur fetch @{} : {}", username, e)
                    results[username] = 0
            await session.commit()

        total = sum(results.values())
        logger.info("Polling terminé. {} nouveaux tweets sur {} comptes.", total, len(usernames))
        return results

    async def run(self) -> None:
        """
        Lance la boucle de polling infinie.
        Respecte l'intervalle défini dans accounts.yaml.
        Appelle d'abord setup() pour synchroniser les comptes.
        """
        self._running = True
        interval = self._config.polling.interval_seconds

        logger.info(
            "Démarrage du collecteur. Intervalle={}s, {} comptes actifs.",
            interval,
            len([a for a in self._config.accounts if a.enabled]),
        )

        await self.sync_accounts_to_db()

        while self._running:
            try:
                await self.poll_once()
            except Exception as e:
                logger.exception("Erreur inattendue dans la boucle de polling : {}", e)

            logger.debug("Prochain polling dans {}s.", interval)
            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Arrête proprement la boucle."""
        self._running = False
        logger.info("Collecteur arrêté.")
