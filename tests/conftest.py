"""
tests/conftest.py
──────────────────
Fixture test_db partagée : DB SQLite temporaire patchée dans tous les modules
qui importent AsyncSessionLocal directement. Disponible dans unit/ et integration/.
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest_asyncio.fixture(scope="function")
async def test_db(tmp_path, monkeypatch):
    """
    Crée une DB SQLite temporaire et patche AsyncSessionLocal partout.
    Restauration automatique via monkeypatch après chaque test.
    """
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test_radar.db"

    test_engine = create_async_engine(
        db_url,
        connect_args={"check_same_thread": False},
    )
    TestSession = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    import src.core.database as db_mod
    from src.core.database import Base

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import src.collector.service as svc_mod
    import src.nlp.processor as nlp_mod
    import src.market.snapshot_scheduler as sched_mod
    import src.alerts.service as alert_mod
    import src.ml.service as ml_mod
    import src.ml.labeler as ml_labeler_mod
    import src.ml.trainer as ml_trainer_mod

    monkeypatch.setattr(db_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(svc_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(nlp_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(sched_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(alert_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(ml_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(ml_labeler_mod, "AsyncSessionLocal", TestSession)
    monkeypatch.setattr(ml_trainer_mod, "AsyncSessionLocal", TestSession)

    yield TestSession

    await test_engine.dispose()
