"""
tests/unit/test_database.py
─────────────────────────────
Régression : SQLite ne conserve pas le tzinfo sur les colonnes DateTime,
ce qui cassait toute comparaison entre une valeur relue en DB et un
datetime.now(tz=timezone.utc) fraîchement créé (SnapshotScheduler, etc.).
AwareDateTime doit garantir un aller-retour tz-aware (UTC).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.core.database import Account, Tweet


class TestAwareDateTime:
    @pytest.mark.asyncio
    async def test_roundtrip_preserves_tz_awareness(self, test_db):
        tweeted_at = datetime.now(tz=timezone.utc)

        async with test_db() as session:
            account = Account(username="tz_trader", markets="[]", tags="[]", priority="medium", enabled=True)
            session.add(account)
            await session.flush()
            session.add(Tweet(
                tweet_id="TZ001", account_id=account.id, text="x",
                tweeted_at=tweeted_at, nlp_processed=False,
            ))
            await session.commit()

        async with test_db() as session:
            tweet = (await session.execute(select(Tweet).where(Tweet.tweet_id == "TZ001"))).scalar_one()

        assert tweet.tweeted_at.tzinfo is not None
        # Comparable à un datetime tz-aware fraîchement créé, sans lever
        # "can't compare offset-naive and offset-aware datetimes".
        now = datetime.now(tz=timezone.utc)
        assert tweet.tweeted_at <= now
        assert abs((tweet.tweeted_at - tweeted_at).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_server_default_timestamp_is_tz_aware(self, test_db):
        # created_at est rempli par server_default=func.now() (CURRENT_TIMESTAMP,
        # UTC sous SQLite) — doit aussi ressortir tz-aware après relecture.
        async with test_db() as session:
            account = Account(username="tz_trader2", markets="[]", tags="[]", priority="medium", enabled=True)
            session.add(account)
            await session.commit()

        async with test_db() as session:
            account = (await session.execute(select(Account).where(Account.username == "tz_trader2"))).scalar_one()

        assert account.created_at.tzinfo is not None
        assert account.created_at <= datetime.now(tz=timezone.utc) + timedelta(seconds=5)
