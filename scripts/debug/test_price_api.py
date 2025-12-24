#!/usr/bin/env python3
"""
Test Price API integration.
Verifies fetching prices from Pokemon TCG API and caching behavior.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.service import PriceService
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def test_price_api():
    """Test price fetching and caching"""
    logger.info("=" * 60)
    logger.info("TESTING PRICE API")
    logger.info("=" * 60)

    # Initialize service
    service = PriceService(cache_ttl=60)  # 1 min TTL for testing

    # Test Card: Charizard Base Set (base1-4)
    card_id = "base1-4"
    
    logger.info(f"\n1. Fetching price for {card_id} (First call - should hit API)...")
    start_time = time.time()
    price = service.get_price(card_id)
    elapsed = (time.time() - start_time) * 1000
    
    if price:
        logger.info(f"  ✓ Success! Time: {elapsed:.1f}ms")
        logger.info(f"  Price: {price.display_price} (Market)")
        logger.info(f"  Source: {price.source}")
        logger.info(f"  URL: {price.url}")
        logger.info(f"  Full Data: {price.to_dict()}")
    else:
        logger.error("  ❌ Failed to fetch price")
        return False

    # Test Caching
    logger.info(f"\n2. Fetching price for {card_id} (Second call - should hit Cache)...")
    start_time = time.time()
    price_cached = service.get_price(card_id)
    elapsed = (time.time() - start_time) * 1000
    
    if price_cached:
        logger.info(f"  ✓ Success! Time: {elapsed:.1f}ms")
        if elapsed < 10:
            logger.info("  ✓ Speed indicates cache hit (<10ms)")
        else:
            logger.warning("  ⚠️  Speed too slow for cache hit")
            
        # Verify data matches
        if price.market == price_cached.market:
             logger.info("  ✓ Data matches original request")
        else:
             logger.error("  ❌ Data mismatch")
    else:
        logger.error("  ❌ Failed to fetch from cache")
        return False

    # Test Invalid Card
    invalid_id = "invalid-card-id-123"
    logger.info(f"\n3. Testing invalid card ID: {invalid_id}...")
    price_invalid = service.get_price(invalid_id)
    
    if price_invalid is None:
        logger.info("  ✓ Correctly returned None for invalid card")
    else:
        logger.error("  ❌ Should have returned None")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("PRICE API TEST COMPLETE - SUCCESS")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    setup_logging(level="INFO")
    success = test_price_api()
    sys.exit(0 if success else 1)
