"""
core/database.py
────────────────
Engine SQLAlchemy async, session factory, et modèles ORM.
SQLite en dev → swap PostgreSQL en prod via DATABASE_URL.
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .settings import get_settings


# ── Engine & Session ──────────────────────────────────────────────────────────

def _make_engine():
    settings = get_settings()
    kwargs = {}
    if "sqlite" in settings.database_url:
        # SQLite ne supporte pas le pool de connexions multithread
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(
        settings.database_url,
        echo=settings.is_dev,   # log SQL en dev
        **kwargs,
    )


engine = _make_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dépendance FastAPI / utilitaire pour obtenir une session async."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Base ORM ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Modèles ───────────────────────────────────────────────────────────────────

class Account(Base):
    """
    Compte Twitter/X surveillé.
    Enrichi progressivement avec le score de fiabilité ML.
    """
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    twitter_id: Mapped[str | None] = mapped_column(String(30), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(100))
    markets: Mapped[str | None] = mapped_column(String(200))    # JSON list
    tags: Mapped[str | None] = mapped_column(String(500))       # JSON list
    priority: Mapped[str] = mapped_column(String(10), default="medium")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Stats calculées (mises à jour périodiquement)
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    profitable_calls: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float | None] = mapped_column(Float)
    avg_roi: Mapped[float | None] = mapped_column(Float)
    reliability_score: Mapped[float | None] = mapped_column(Float)  # 0-1, ML

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    tweets: Mapped[list["Tweet"]] = relationship(back_populates="account")

    def __repr__(self) -> str:
        return f"<Account @{self.username} score={self.reliability_score}>"


class Tweet(Base):
    """
    Tweet brut + données enrichies (NLP + marché).
    """
    __tablename__ = "tweets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tweet_id: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)

    # Contenu brut
    text: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[str | None] = mapped_column(String(5))
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    retweet_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    tweeted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Données NLP (remplies par Module 3)
    tickers: Mapped[str | None] = mapped_column(String(500))    # JSON ["BTC","ETH"]
    call_type: Mapped[str | None] = mapped_column(String(20))   # long|short|hold|info|unclear
    sentiment: Mapped[float | None] = mapped_column(Float)      # -1.0 à 1.0
    confidence: Mapped[float | None] = mapped_column(Float)     # score NLP 0-1
    target_price: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    urgency_score: Mapped[float | None] = mapped_column(Float)  # 0-1

    # Score ML final (Module 4)
    ml_score: Mapped[float | None] = mapped_column(Float)       # 0-1

    # Metadata
    is_retweet: Mapped[bool] = mapped_column(Boolean, default=False)
    nlp_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    account: Mapped["Account"] = relationship(back_populates="tweets")
    market_snapshots: Mapped[list["MarketSnapshot"]] = relationship(back_populates="tweet")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="tweet")

    __table_args__ = (
        Index("ix_tweets_tweeted_at", "tweeted_at"),
        Index("ix_tweets_account_id", "account_id"),
        Index("ix_tweets_nlp_processed", "nlp_processed"),
        Index("ix_tweets_alerted_at", "alerted_at"),
    )

    def __repr__(self) -> str:
        return f"<Tweet {self.tweet_id} @{self.account_id} {self.call_type}>"


class MarketSnapshot(Base):
    """
    Prix d'un ticker au moment du tweet + snapshots futurs.
    Permet de mesurer la performance réelle de chaque call.
    """
    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tweet_id: Mapped[int] = mapped_column(ForeignKey("tweets.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    market_type: Mapped[str] = mapped_column(String(10))        # crypto | stock

    # Prix au moment du tweet (t=0)
    price_at_tweet: Mapped[float | None] = mapped_column(Float)
    volume_at_tweet: Mapped[float | None] = mapped_column(Float)
    snapshot_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Snapshots futurs (remplis progressivement par le scheduler)
    price_1h: Mapped[float | None] = mapped_column(Float)
    price_4h: Mapped[float | None] = mapped_column(Float)
    price_24h: Mapped[float | None] = mapped_column(Float)
    price_7d: Mapped[float | None] = mapped_column(Float)

    # Variations calculées (%)
    change_1h: Mapped[float | None] = mapped_column(Float)
    change_4h: Mapped[float | None] = mapped_column(Float)
    change_24h: Mapped[float | None] = mapped_column(Float)
    change_7d: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    tweet: Mapped["Tweet"] = relationship(back_populates="market_snapshots")

    __table_args__ = (
        Index("ix_snapshots_tweet_ticker", "tweet_id", "ticker"),
    )


class Prediction(Base):
    """
    Prédiction ML pour un tweet + résultat réel (pour le backtest).
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tweet_id: Mapped[int] = mapped_column(ForeignKey("tweets.id"), nullable=False)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_profitable: Mapped[bool | None] = mapped_column(Boolean)
    predicted_change_pct: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)

    # Rempli après le fait
    actual_profitable: Mapped[bool | None] = mapped_column(Boolean)
    actual_change_pct: Mapped[float | None] = mapped_column(Float)
    evaluation_window: Mapped[str | None] = mapped_column(String(10))  # 1h|4h|24h|7d

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    tweet: Mapped["Tweet"] = relationship(back_populates="predictions")
