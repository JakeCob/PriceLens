"""
Price Service
Orchestrates price fetching, caching, source management, and trend tracking.
"""

import logging
from typing import Optional, Tuple

from src.api.base import PriceData, PriceSource
from src.api.base import PriceData, PriceSource
from src.api.history import PriceHistoryRepository, price_to_dict_with_trend
from src.api.pokemontcg_client import PokemonTCGClient
from src.api.custom_source import CustomPriceSource
from src.api.smart_cache import SmartCache

logger = logging.getLogger(__name__)


class PriceService:
    """
    High-level service for fetching card prices.
    Combines API client, smart caching, and history tracking.
    """

    def __init__(self, enable_history: bool = True):
        """
        Initialize price service
        
        Args:
            enable_history: Whether to persist price history and compute trends
        """
        self.cache = SmartCache()
        
        # Sources: Custom (Priority) -> API (Fallback)
        self.custom_source = CustomPriceSource()
        self.api_source = PokemonTCGClient()
        
        self.history = PriceHistoryRepository() if enable_history else None
        
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

        # 2. Check Custom Source (Fast, Local)
        try:
            price = self.custom_source.get_price(card_id)
            if price:
                logger.debug(f"Found custom price for {card_id}")
                return price
        except Exception as e:
            logger.warning(f"Custom source error: {e}")

        # 3. Fetch from API (Slow, Remote)
        try:
            price = self.api_source.get_price(card_id)
            
            if price:
                # 4. Update cache
                self.cache.set(card_id, price)

                # 5. Persist history for trend tracking
                if self.history:
                    try:
                        self.history.record(card_id, price)
                    except Exception as history_err:
                        logger.warning(f"Could not record price history: {history_err}")

                return price
                
        except Exception as e:
            logger.error(f"Error fetching price from API: {e}")

        return None

    def get_price_with_trend(self, card_id: str) -> Tuple[Optional[PriceData], Optional[float]]:
        """
        Get price for a card and return the day-over-day (last snapshot) trend percentage.
        """
        price = self.get_price(card_id)
        if not price or not self.history:
            return price, None

        try:
            trend_percent, _, _ = self.history.get_trend(card_id)
        except Exception as e:
            logger.warning(f"Could not compute price trend: {e}")
            trend_percent = None
        return price, trend_percent

    def get_price_dict(self, card_id: str) -> dict:
        """
        Convenience helper for API responses: returns a dict with trend info.
        """
        price, trend = self.get_price_with_trend(card_id)
        return price_to_dict_with_trend(card_id, price, trend)
