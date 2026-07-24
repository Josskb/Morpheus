#!/usr/bin/env python3
"""
Launch Trade Tracker Dashboard
Accessible via http://192.168.1.67:5000
"""

from src.tracking.dashboard_api import create_app
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting Trade Tracker Dashboard...")
    logger.info("Accessible at: http://192.168.1.67:5000")

    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
