"""
Dashboard API: Simple Flask server for trade tracking UI
"""

from __future__ import annotations

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timezone
import json

from loguru import logger
from .trade_tracker import TradeTracker


def create_app() -> Flask:
    """Create Flask app with routes."""
    app = Flask(__name__, template_folder="templates")
    tracker = TradeTracker()

    @app.route("/")
    def dashboard():
        """Render dashboard HTML."""
        return render_template("dashboard.html")

    @app.route("/api/alerts")
    def get_alerts():
        """Get recent alerts."""
        try:
            alerts = tracker.get_recent_alerts(limit=20)
            return jsonify({"success": True, "alerts": alerts})
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trades")
    def get_trades():
        """Get trades."""
        try:
            status = request.args.get("status")
            trades = tracker.get_trades(status=status)
            return jsonify({"success": True, "trades": trades})
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/stats")
    def get_stats():
        """Get trading statistics."""
        try:
            stats = tracker.get_stats()
            return jsonify({"success": True, "stats": stats})
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trades", methods=["POST"])
    def create_trade():
        """Accept an alert and create a trade."""
        try:
            data = request.json
            alert_id = data.get("alert_id")
            entry_price = float(data.get("entry_price", 0))
            target_price = data.get("target_price")
            stop_loss = data.get("stop_loss")
            notes = data.get("notes", "")

            if target_price:
                target_price = float(target_price)
            if stop_loss:
                stop_loss = float(stop_loss)

            trade_id = tracker.accept_alert(
                alert_id=alert_id,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                notes=notes,
            )

            return jsonify({"success": True, "trade_id": trade_id})
        except Exception as e:
            logger.error(f"Error creating trade: {e}")
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/trades/<int:trade_id>", methods=["PUT"])
    def update_trade(trade_id: int):
        """Update trade with current price."""
        try:
            data = request.json
            current_price = float(data.get("current_price", 0))
            notes = data.get("notes")

            tracker.update_trade(trade_id, current_price, notes=notes)

            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/trades/<int:trade_id>/close", methods=["POST"])
    def close_trade(trade_id: int):
        """Close a trade."""
        try:
            data = request.json
            final_price = float(data.get("final_price", 0))
            notes = data.get("notes", "")

            tracker.close_trade(trade_id, final_price, notes=notes)

            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return jsonify({"success": False, "error": str(e)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
