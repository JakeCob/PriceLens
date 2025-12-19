"""
Price Service
Orchestrates price fetching, caching, source management, and trend tracking.

Fallback Chain:
1. SmartCache (fresh) -> hit? return
2. Custom Source (local JSON) -> hit? return
3. Check if recently failed -> yes? return stale
4. Pokemon TCG API -> success? cache & return
5. TCGPlayer Scraper (fallback) -> success? cache & return
6. SmartCache (stale) -> hit? return old price
7. Return None
"""

import logging
from typing import Optional, Tuple

from src.api.base import PriceData, PriceSource
from src.api.errors import PermanentError, TransientError, CircuitOpenError, RateLimitError
from src.api.history import PriceHistoryRepository, price_to_dict_with_trend
from src.api.pokemontcg_client import PokemonTCGClient
from src.api.custom_source import CustomPriceSource
from src.api.smart_cache import SmartCache

logger = logging.getLogger(__name__)


class PriceService:
    """
    High-level service for fetching card prices.
    Combines API client, smart caching, fallback sources, and history tracking.

    Handles transient failures gracefully by:
    - Returning stale cached data during outages
    - Caching failures with short TTL to avoid repeated hammering
    - Using fallback sources (scraper) when primary API fails
    """

    def __init__(self, enable_history: bool = True):
        """
        Initialize price service

        Args:
            enable_history: Whether to persist price history and compute trends
        """
        self.cache = SmartCache()

        # Sources: Custom (Priority) -> API (Primary) -> Scraper (Fallback)
        self.custom_source = CustomPriceSource()
        self.api_source = PokemonTCGClient()

        # Fallback scraper (lazy-loaded to avoid import if not needed)
        self._scraper = None

        self.history = PriceHistoryRepository() if enable_history else None

        logger.info("PriceService initialized")

    @property
    def scraper(self):
        """Lazy-load TCGPlayer scraper"""
        if self._scraper is None:
            try:
                from src.api.tcgplayer_scraper import TCGPlayerScraper
                self._scraper = TCGPlayerScraper()
                logger.info("TCGPlayer scraper initialized as fallback")
            except ImportError as e:
                logger.warning(f"TCGPlayer scraper not available: {e}")
                self._scraper = False  # Mark as unavailable
        return self._scraper if self._scraper else None

    def get_price(self, card_id: str, timeout_seconds: Optional[float] = None) -> Optional[PriceData]:
        """
        Get price for a card with full fallback chain.

        Args:
            card_id: Card ID
            timeout_seconds: Optional timeout for remote API fetch

        Returns:
            PriceData or None
        """
        # 1. Check fresh cache
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

        # 3. Check if recently failed (avoid hammering)
        if self.cache.is_recently_failed(card_id):
            logger.debug(f"Skipping API for recently failed {card_id}, returning stale")
            return self.cache.get_stale(card_id)

        # 4. Try primary API with retry
        tcgplayer_url = None  # Save for scraper fallback
        try:
            price = self.api_source.get_price(card_id, timeout_seconds=timeout_seconds)

            if price:
                self._on_price_success(card_id, price)
                return price

        except PermanentError as e:
            # Card doesn't exist in API - cache None for 24h
            logger.info(f"Card {card_id} not found in API: {e}")
            self.cache.set(card_id, None)
            return None

        except (TransientError, RateLimitError, CircuitOpenError) as e:
            # Temporary failure - try fallback sources
            logger.warning(f"Transient error for {card_id}: {e}")
            tcgplayer_url = self._get_tcgplayer_url_from_cache(card_id)

        except Exception as e:
            logger.error(f"Unexpected error fetching {card_id}: {e}")

        # 5. Try scraper fallback (if available and we have a URL)
        if self.scraper and tcgplayer_url:
            try:
                price = self.scraper.get_price_from_url(tcgplayer_url)
                if price:
                    logger.info(f"Got price for {card_id} from scraper fallback")
                    self._on_price_success(card_id, price)
                    return price
            except Exception as e:
                logger.warning(f"Scraper fallback failed for {card_id}: {e}")

        # 6. Cache the failure and return stale data
        self.cache.set_failure(card_id, "API and fallback failed")
        stale_price = self.cache.get_stale(card_id)

        if stale_price:
            logger.info(f"Returning stale price for {card_id}")
            return stale_price

        # 7. No data available
        return None

    def _on_price_success(self, card_id: str, price: PriceData) -> None:
        """Handle successful price fetch"""
        # Update cache
        self.cache.set(card_id, price)
        self.cache.clear_failure(card_id)

        # Persist history for trend tracking
        if self.history:
            try:
                self.history.record(card_id, price)
            except Exception as history_err:
                logger.warning(f"Could not record price history: {history_err}")

    def _get_tcgplayer_url_from_cache(self, card_id: str) -> Optional[str]:
        """Try to get TCGPlayer URL from stale cache for scraper fallback"""
        stale = self.cache.get_stale(card_id)
        if stale and hasattr(stale, 'url') and stale.url:
            return stale.url
        return None

    def get_price_with_trend(self, card_id: str) -> Tuple[Optional[PriceData], Optional[float]]:
        """
        Get price for a card and return the day-over-day trend percentage.
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

    def get_status(self) -> dict:
        """Get service status including circuit breaker and failure stats"""
        return {
            "api_circuit": self.api_source.get_circuit_status(),
            "failure_cache": self.cache.get_failure_stats(),
            "scraper_available": self.scraper is not None,
        }
