"""Logging configuration for PriceLens"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    name: str = "pricelens",
    log_dir: str = "logs",
    level: int = logging.DEBUG,  # Changed to DEBUG
    log_file: bool = True  # Renamed back to log_file
) -> logging.Logger:
    """
    Configure logging for the application

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to also log to file
    """
    # Create logs directory if needed
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"pricelens_{timestamp}.log"
        file_handler = logging.FileHandler(log_path)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Suppress verbose library logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")
    if log_file:
        logger.info(f"Log file: {log_path}")

    return logger