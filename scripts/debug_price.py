"""
Debug script to test price fetching for a specific card.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.api.service import PriceService

setup_logging(level="DEBUG")
logger = logging.getLogger("debug_price")

def test_price_fetch(card_id: str):
    logger.info(f"Testing price fetch for: {card_id}")
    
    service = PriceService()
    
    # Force fetch (bypass cache if possible, though service checks cache first)
    # We can inspect the service's primary source directly to isolate API
    
    logger.info("1. Testing API Source (Direct)...")
    try:
        # Access api_source directly (was primary_source)
        if hasattr(service, 'api_source'):
            price = service.api_source.get_price(card_id)
        else:
            price = service.primary_source.get_price(card_id)
            
        if price:
            logger.info(f"✅ API Success: {price.to_dict()}")
        else:
            logger.error("❌ API returned None")
    except Exception as e:
        logger.error(f"❌ API Exception: {e}")

    logger.info("2. Testing Service (with Cache)...")
    try:
        price = service.get_price(card_id)
        if price:
            logger.info(f"✅ Service Success: {price.to_dict()}")
        else:
            logger.error("❌ Service returned None")
    except Exception as e:
        logger.error(f"❌ Service Exception: {e}")

if __name__ == "__main__":
    card_id = sys.argv[1] if len(sys.argv) > 1 else "base1-4"
    test_price_fetch(card_id)
