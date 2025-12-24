#!/usr/bin/env python3
"""
PriceLens API Server Launcher
Starts the FastAPI backend server.
"""

import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(level="DEBUG")
    
    logger.info("Starting PriceLens API Server...")
    logger.info("Open http://localhost:7848 in your browser")
    logger.info("(If running remotely, use your server's IP or Proxy URL)")
    
    # Start Server
    try:
        uvicorn.run(
            "src.web.api:app",
            host="0.0.0.0",
            port=7848,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
