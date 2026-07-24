"""
Trade Tracker: Records and monitors actual trading results from alerts
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
import sqlite3
import json

from loguru import logger


class TradeRecord(NamedTuple):
    id: int
    alert_id: str
    username: str
    ticker: str
    entry_price: float | None
    target_price: float | None
    stop_loss: float | None
    status: str  # pending, active, closed_profit, closed_loss, closed_neutral
    result_pct: float | None
    notes: str | None
    alert_time: str
    trade_time: str | None
    close_time: str | None


class TradeTracker:
    """Track alert outcomes and trading results."""

    DB_PATH = Path(__file__).parent.parent.parent / "data" / "trades.db"

    def __init__(self):
        self.db_path = self.DB_PATH
        self._init_db()

    def _init_db(self):
        """Initialize trades database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                ticker TEXT NOT NULL,
                nlp_score REAL,
                ml_score REAL,
                hybrid_score REAL,
                ml_profitable BOOLEAN,
                account_reliability REAL,
                account_win_rate REAL,
                alert_message TEXT,
                alert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                ticker TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                status TEXT DEFAULT 'pending',
                result_pct REAL,
                notes TEXT,
                alert_time TIMESTAMP,
                trade_time TIMESTAMP,
                close_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (alert_id) REFERENCES trade_alerts(alert_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                price REAL,
                status TEXT,
                notes TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        conn.commit()
        conn.close()

        logger.info(f"Trade database ready: {self.db_path}")

    def log_alert(
        self,
        alert_id: str,
        username: str,
        ticker: str,
        nlp_score: float,
        ml_score: float,
        hybrid_score: float,
        ml_profitable: bool,
        account_reliability: float,
        account_win_rate: float,
        message: str,
    ) -> None:
        """Log an alert from AlertService."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trade_alerts
                (alert_id, username, ticker, nlp_score, ml_score, hybrid_score,
                 ml_profitable, account_reliability, account_win_rate, alert_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id, username, ticker, nlp_score, ml_score, hybrid_score,
                ml_profitable, account_reliability, account_win_rate, message
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log alert {alert_id}: {e}")
        finally:
            conn.close()

    def accept_alert(
        self,
        alert_id: str,
        entry_price: float,
        target_price: float | None = None,
        stop_loss: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Accept an alert and create a trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get alert details
            cursor.execute("""
                SELECT username, ticker, alert_time FROM trade_alerts WHERE alert_id = ?
            """, (alert_id,))
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Alert {alert_id} not found")

            username, ticker, alert_time = row

            # Create trade
            cursor.execute("""
                INSERT INTO trades
                (alert_id, username, ticker, entry_price, target_price, stop_loss,
                 status, trade_time, alert_time, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id, username, ticker, entry_price, target_price, stop_loss,
                "active", datetime.now(tz=timezone.utc).isoformat(), alert_time, notes
            ))
            trade_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Trade {trade_id} created for {username}/{ticker} @ {entry_price}")
            return trade_id

        except Exception as e:
            logger.error(f"Failed to accept alert {alert_id}: {e}")
            raise
        finally:
            conn.close()

    def update_trade(
        self,
        trade_id: int,
        current_price: float,
        status: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Update trade with current price and status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get trade entry price
            cursor.execute("SELECT entry_price, target_price, stop_loss FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Trade {trade_id} not found")

            entry, target, stop = row
            result_pct = ((current_price - entry) / entry) * 100 if entry else 0

            # Auto-determine status if not provided
            if status is None:
                if target and current_price >= target:
                    status = "closed_profit"
                elif stop and current_price <= stop:
                    status = "closed_loss"
                elif current_price > entry:
                    status = "active_winning"
                else:
                    status = "active_losing"

            # Update trade
            cursor.execute("""
                UPDATE trades
                SET result_pct = ?, status = ?
                WHERE id = ?
            """, (result_pct, status, trade_id))

            # Log update
            cursor.execute("""
                INSERT INTO trade_updates (trade_id, price, status, notes)
                VALUES (?, ?, ?, ?)
            """, (trade_id, current_price, status, notes))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update trade {trade_id}: {e}")
        finally:
            conn.close()

    def close_trade(self, trade_id: int, final_price: float, notes: str = "") -> None:
        """Close a trade with final price."""
        self.update_trade(trade_id, final_price, status="closed", notes=notes)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades SET close_time = ? WHERE id = ?
        """, (datetime.now(tz=timezone.utc).isoformat(), trade_id))
        conn.commit()
        conn.close()

    def get_recent_alerts(self, limit: int = 10) -> list[dict]:
        """Get recent alerts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM trade_alerts
            ORDER BY alert_time DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_trades(self, status: str | None = None) -> list[dict]:
        """Get trades, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM trades WHERE status = ? ORDER BY trade_time DESC
            """, (status,))
        else:
            cursor.execute("SELECT * FROM trades ORDER BY trade_time DESC")

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get trading statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status LIKE 'closed%'")
        total_closed = cursor.fetchone()[0]

        # Winning trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed_profit'")
        total_wins = cursor.fetchone()[0]

        # Losing trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed_loss'")
        total_losses = cursor.fetchone()[0]

        # Average result
        cursor.execute("SELECT AVG(result_pct) FROM trades WHERE status LIKE 'closed%'")
        avg_result = cursor.fetchone()[0] or 0

        # Total ROI
        cursor.execute("SELECT SUM(result_pct) FROM trades WHERE status LIKE 'closed%'")
        total_roi = cursor.fetchone()[0] or 0

        conn.close()

        win_rate = (total_wins / total_closed * 100) if total_closed > 0 else 0

        return {
            "total_trades": total_closed,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": win_rate,
            "avg_result_pct": avg_result,
            "total_roi_pct": total_roi,
        }
