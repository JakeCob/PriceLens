"""
Price Service
Orchestrates price fetching, caching, and source management.
"""

import logging
from typing import Optional

from src.api.base import PriceData, PriceSource
from src.api.cache import PriceCache
from src.api.pokemontcg_client import PokemonTCGClient

logger = logging.getLogger(__name__)


class PriceService:
    """
    Service to get card prices.
    Handles caching and source selection.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize price service
        
        Args:
            cache_ttl: Cache TTL in seconds
        """
        self.cache = PriceCache(ttl_seconds=cache_ttl)
        
        # Initialize sources (currently only Pokemon TCG API)
        # In future, we could add more sources and fallback logic
        self.primary_source: PriceSource = PokemonTCGClient()
        
        logger.info("PriceService initialized")

    def get_price(self, card_id: str) -> Optional[PriceData]:
        """
        Get price for a card (checks cache first)
        
        Args:
            card_id: Card ID
            
        Returns:
            PriceData or None
        """
        # 1. Check cache
        cached_price = self.cache.get(card_id)
        if cached_price:
            return cached_price

        # 2. Fetch from primary source
        try:
            price = self.primary_source.get_price(card_id)
            
            if price:
                # 3. Update cache
                self.cache.set(card_id, price)
                return price
                
        except Exception as e:
            logger.error(f"Error fetching price from primary source: {e}")

        return None
