"""
Caching system for Price API.
"""

import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.api.base import PriceData

logger = logging.getLogger(__name__)


class PriceCache:
    """
    File-based cache for price data with TTL support.
    """

    def __init__(self, cache_file: str = "data/cache/prices.pkl", ttl_seconds: int = 300):
        """
        Initialize cache
        
        Args:
            cache_file: Path to cache file
            ttl_seconds: Time to live in seconds (default 5 mins)
        """
        self.cache_file = Path(cache_file)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Dict[str, Tuple[PriceData, datetime]] = {}
        
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} items from price cache")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, card_id: str) -> Optional[PriceData]:
        """
        Get price from cache if valid
        
        Args:
            card_id: Card ID
            
        Returns:
            PriceData if found and valid, else None
        """
        if card_id not in self.cache:
            return None

        data, timestamp = self.cache[card_id]
        
        # Check TTL
        if datetime.now() - timestamp > self.ttl:
            logger.debug(f"Cache expired for {card_id}")
            del self.cache[card_id]
            return None

        logger.debug(f"Cache hit for {card_id}")
        return data

    def set(self, card_id: str, data: PriceData) -> None:
        """
        Set price in cache
        
        Args:
            card_id: Card ID
            data: PriceData object
        """
        self.cache[card_id] = (data, datetime.now())
        self._save_cache()
