#!/usr/bin/env python3
"""
PriceLens Web Interface Launcher
Starts the FastAPI server and opens the browser.
"""

import uvicorn
import logging
import webbrowser
import threading
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8080")
    except Exception:
        pass  # Ignore if browser cannot be opened

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(level="DEBUG")
    
    logger.info("Starting PriceLens Web Server...")
    logger.info("Open http://localhost:8080 in your browser")
    logger.info("(If running remotely, use your server's IP or Proxy URL)")
    
    # Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Server
    try:
        uvicorn.run(
            "src.web.api:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
