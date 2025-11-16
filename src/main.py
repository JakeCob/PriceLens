#!/usr/bin/env python3
"""Main entry point for PriceLens application"""

import sys
import argparse
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils.logging_config import setup_logging


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='PriceLens - Pokemon Card Price Overlay System'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=None,
        help='Camera source (override config)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(level=log_level)

    # Load configuration
    try:
        config = Config(args.config)
        logger.info("Configuration loaded")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override configuration with command line arguments
    if args.camera is not None:
        config.config['camera']['source'] = args.camera
    if args.no_gpu:
        config.config['performance']['use_gpu'] = False

    # Display startup information
    logger.info("=" * 60)
    logger.info("PriceLens - Pokemon Card Price Overlay System")
    logger.info(f"Version: {config.get('app.version', '0.1.0')}")
    logger.info(f"Camera: {config.get('camera.source')}")
    logger.info(f"GPU: {'Enabled' if config.get('performance.use_gpu') else 'Disabled'}")
    logger.info("=" * 60)

    # Check for required files
    model_path = Path(config.get('detection.model_path'))
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run 'python scripts/download_models.py' to download required models")
        sys.exit(1)

    # TODO: Initialize and run main application
    logger.info("Application components will be initialized here...")
    logger.info("Press Ctrl+C to exit")

    try:
        # Placeholder for main application loop
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    logger.info("Application terminated")


if __name__ == "__main__":
    main()