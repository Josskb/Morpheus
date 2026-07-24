#!/usr/bin/env python3
"""
Populate test data for dashboard visualization
Creates realistic trade history for demo purposes
"""

from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path
import random

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trades.db"

def populate_test_data():
    """Add realistic test trades to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Sample traders
    traders = [
        ("DeItaone", 0.82, 0.62),
        ("glassnode", 0.71, 0.58),
        ("SPY", 0.68, 0.55),
        ("MSFT", 0.65, 0.52),
        ("XRP", 0.70, 0.60),
        ("ETH", 0.72, 0.59),
    ]

    tickers = ["BTC", "SOL", "ETH", "XRP", "ADA", "DOGE", "AVAX", "MATIC"]

    base_time = datetime.now(tz=timezone.utc) - timedelta(days=7)

    alerts = []
    trades = []

    # Create 20 past alerts
    for i in range(20):
        trader, nlp_score, win_rate = random.choice(traders)
        ticker = random.choice(tickers)
        ml_score = random.uniform(0.45, 0.80)
        hybrid_score = (0.65 * nlp_score) + (0.35 * ml_score)

        alert_time = base_time + timedelta(hours=i * 8)
        alert_id = f"alert_{i}_{int(alert_time.timestamp())}"

        cursor.execute("""
            INSERT OR IGNORE INTO trade_alerts
            (alert_id, username, ticker, nlp_score, ml_score, hybrid_score,
             ml_profitable, account_reliability, account_win_rate, alert_message, alert_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_id, trader, ticker, nlp_score, ml_score, hybrid_score,
            random.choice([0, 1]), 0.8 if "DeItaone" in trader else 0.6,
            win_rate, f"@{trader} | {ticker}", alert_time.isoformat()
        ))

        alerts.append({
            "id": i,
            "alert_id": alert_id,
            "alert_time": alert_time,
            "trader": trader,
            "ticker": ticker,
            "hybrid_score": hybrid_score,
        })

    # Create 15 past trades (mix of wins and losses)
    wins = 0
    losses = 0
    total_pnl = 0

    for i in range(15):
        alert = alerts[i]
        entry_price = random.uniform(20000, 50000)

        # Generate realistic P&L distribution
        # 60% should be wins, 40% losses, with varying %
        is_win = random.random() < 0.60
        if is_win:
            result_pct = random.uniform(0.5, 8.0)
            status = "closed_profit"
            wins += 1
        else:
            result_pct = random.uniform(-5.0, -0.5)
            status = "closed_loss"
            losses += 1

        total_pnl += result_pct

        final_price = entry_price * (1 + result_pct / 100)
        trade_time = alert["alert_time"] + timedelta(minutes=random.randint(5, 60))
        close_time = trade_time + timedelta(hours=random.randint(1, 24))

        cursor.execute("""
            INSERT INTO trades
            (alert_id, username, ticker, entry_price, target_price, stop_loss,
             status, result_pct, notes, alert_time, trade_time, close_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert["alert_id"],
            alert["trader"],
            alert["ticker"],
            round(entry_price, 2),
            round(entry_price * 1.05, 2),  # 5% target
            round(entry_price * 0.95, 2),  # 5% stop
            status,
            round(result_pct, 2),
            "Test trade" if random.random() > 0.7 else "",
            alert["alert_time"].isoformat(),
            trade_time.isoformat(),
            close_time.isoformat(),
        ))

    conn.commit()
    conn.close()

    print("✅ Test data populated!")
    print(f"   • 20 alerts created")
    print(f"   • 15 trades created")
    print(f"   • {wins} wins, {losses} losses")
    print(f"   • Total ROI: {total_pnl:+.2f}%")
    print(f"   • Win rate: {wins / 15 * 100:.1f}%")

if __name__ == "__main__":
    populate_test_data()
