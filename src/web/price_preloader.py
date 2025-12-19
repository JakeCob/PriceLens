"""
Price Preloader - Batch loads prices for all known cards
Runs on startup and refreshes periodically

Features:
- Stale-while-revalidate: loads old prices immediately, refreshes in background
- Failed card retry: retries failed cards at end of batch
- Partial completion tracking: knows when refresh was incomplete
"""
import os
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime, timedelta

from src.api.service import PriceService

logger = logging.getLogger(__name__)


class PricePreloader:
    """
    Preloads prices for all known cards at startup
    Refreshes periodically to keep prices fresh
    """
    
    def __init__(
        self,
        price_service: PriceService,
        refresh_interval_hours: int = 1,
        *,
        eager_on_start: bool | None = None,
        startup_delay_seconds: float | None = None,
        batch_size: int | None = None,
        batch_delay_seconds: float | None = None,
    ):
        """
        Initialize price preloader
        
        Args:
            price_service: PriceService instance
            refresh_interval_hours: Auto-refresh interval (default: 1 hour)
            eager_on_start: If True, run refresh immediately during startup (prices ready sooner, higher load)
            startup_delay_seconds: Delay before starting the first refresh (when not eager)
            batch_size: Number of card IDs to fetch concurrently per batch
            batch_delay_seconds: Sleep between batches (throttle)
        """
        self.service = price_service
        self.refresh_interval = timedelta(hours=refresh_interval_hours)
        self.cache: Dict[str, Dict] = {}  # card_id -> {price, timestamp}
        self.last_refresh = None
        self.refresh_task = None
        self.is_refreshing = False
        self.progress = {"current": 0, "total": 0, "status": "idle"}

        # Runtime knobs (env overrides)
        def _env_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}

        def _env_float(name: str, default: float) -> float:
            v = os.getenv(name)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _env_int(name: str, default: int) -> int:
            v = os.getenv(name)
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        self.eager_on_start = eager_on_start if eager_on_start is not None else _env_bool("PRICE_PRELOAD_EAGER", False)
        self.startup_delay_seconds = startup_delay_seconds if startup_delay_seconds is not None else _env_float("PRICE_PRELOAD_STARTUP_DELAY_SECONDS", 2.0)
        self.batch_size = batch_size if batch_size is not None else _env_int("PRICE_PRELOAD_BATCH_SIZE", 10)
        self.batch_delay_seconds = batch_delay_seconds if batch_delay_seconds is not None else _env_float("PRICE_PRELOAD_BATCH_DELAY_SECONDS", 0.2)

        # Retry configuration
        self.retry_delay_seconds = _env_float("PRICE_PRELOAD_RETRY_DELAY_SECONDS", 30.0)
        self.retry_per_card_delay = _env_float("PRICE_PRELOAD_RETRY_PER_CARD_DELAY", 1.0)
        
    async def start(self):
        """Start preloader with initial load"""
        logger.info("Starting price preloader...")
        
        # 1. Load stale prices immediately (Stale-While-Revalidate)
        await self.load_stale_prices()
        
        # 2. Start fresh load
        if self.eager_on_start:
            logger.info("PRICE_PRELOAD_EAGER enabled: refreshing prices immediately on startup")
            await self.refresh_all_prices()
        else:
            asyncio.create_task(self.delayed_refresh())
        
        # 3. Start periodic refresh task
        self.refresh_task = asyncio.create_task(self._periodic_refresh())

    async def delayed_refresh(self):
        """Wait for system to stabilize before hammering API"""
        logger.info(f"Waiting {self.startup_delay_seconds:.1f}s before starting background price refresh...")
        await asyncio.sleep(self.startup_delay_seconds)
        await self.refresh_all_prices()

    async def load_stale_prices(self):
        """Load stale prices from persistent cache immediately"""
        try:
            logger.info("Loading stale prices from cache...")
            card_ids = self._get_all_card_ids()
            count = 0
            for card_id in card_ids:
                # Use get_stale to retrieve expired entries from SmartCache
                # We access the underlying cache directly to bypass TTL checks
                price_data = self.service.cache.get_stale(card_id)
                
                if price_data:
                    self.cache[card_id] = {
                        'price': price_data.market,
                        'timestamp': datetime.now().timestamp(),
                        'data': price_data.to_dict()
                    }
                    count += 1
            logger.info(f"Loaded {count} stale prices from cache")
        except Exception as e:
            logger.error(f"Error loading stale prices: {e}")
        
    async def stop(self):
        """Stop periodic refresh"""
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Price preloader stopped")
        
    async def _periodic_refresh(self):
        """Periodically refresh prices"""
        while True:
            try:
                await asyncio.sleep(self.refresh_interval.total_seconds())
                logger.info("Starting periodic price refresh...")
                await self.refresh_all_prices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic refresh error: {e}")
                
    async def refresh_all_prices(self):
        """Load prices for all known cards with retry for failures"""
        if self.is_refreshing:
            logger.warning("Refresh already in progress")
            return

        self.is_refreshing = True
        self.progress["status"] = "loading"
        self.progress["failed_cards"] = 0
        self.progress["retried_cards"] = 0

        try:
            # Get all card IDs from feature databases
            card_ids = self._get_all_card_ids()
            total = len(card_ids)

            logger.info(f"Starting price refresh for {total} cards...")
            self.progress["total"] = total
            self.progress["current"] = 0

            # Fetch prices in batches to avoid overwhelming the API
            batch_size = max(1, int(self.batch_size))
            batch_delay = max(0.0, float(self.batch_delay_seconds))

            failed_cards: List[str] = []
            successful_count = 0

            for i in range(0, total, batch_size):
                batch = card_ids[i:i + batch_size]

                # Fetch batch concurrently, tracking success/failure
                tasks = [
                    asyncio.to_thread(self._fetch_and_cache_with_status, card_id)
                    for card_id in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Track failures
                for card_id, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed_cards.append(card_id)
                        logger.debug(f"Batch fetch exception for {card_id}: {result}")
                    elif result is False:  # _fetch_and_cache_with_status returns False on failure
                        failed_cards.append(card_id)
                    else:
                        successful_count += 1

                self.progress["current"] = min(i + batch_size, total)

                # Delay between batches to avoid saturating network / CPU
                if batch_delay:
                    await asyncio.sleep(batch_delay)

            # Retry failed cards with longer delays
            if failed_cards:
                logger.info(f"Retrying {len(failed_cards)} failed cards after {self.retry_delay_seconds}s...")
                self.progress["status"] = "retrying"
                self.progress["failed_cards"] = len(failed_cards)
                await asyncio.sleep(self.retry_delay_seconds)

                retry_success = 0
                for card_id in failed_cards:
                    try:
                        result = await asyncio.to_thread(self._fetch_and_cache_with_status, card_id)
                        if result is True:
                            retry_success += 1
                    except Exception as e:
                        logger.debug(f"Retry failed for {card_id}: {e}")

                    # Delay between individual retries
                    await asyncio.sleep(self.retry_per_card_delay)

                self.progress["retried_cards"] = retry_success
                final_failed = len(failed_cards) - retry_success
                self.progress["failed_cards"] = final_failed
                logger.info(f"Retry complete: {retry_success}/{len(failed_cards)} succeeded")

            # Determine final status
            self.last_refresh = datetime.now()
            final_failed = self.progress.get("failed_cards", 0)
            success_rate = (total - final_failed) / total if total > 0 else 1.0

            if final_failed == 0:
                self.progress["status"] = "complete"
            elif success_rate >= 0.5:
                self.progress["status"] = "partial"
                logger.warning(f"Refresh partially complete: {final_failed}/{total} cards failed")
            else:
                self.progress["status"] = "mostly_failed"
                logger.error(f"Refresh mostly failed: {final_failed}/{total} cards failed")

            logger.info(f"Price refresh done. Cached {len(self.cache)} prices, {final_failed} failed.")

        except Exception as e:
            logger.error(f"Error refreshing prices: {e}")
            self.progress["status"] = "error"
        finally:
            self.is_refreshing = False
            
    def _fetch_and_cache(self, card_id: str):
        """Fetch and cache a single card's price (legacy, doesn't return status)"""
        self._fetch_and_cache_with_status(card_id)

    def _fetch_and_cache_with_status(self, card_id: str) -> bool:
        """
        Fetch and cache a single card's price.

        Returns:
            True if price was successfully fetched
            False if fetch failed (for retry tracking)
        """
        try:
            price_data = self.service.get_price(card_id)

            if price_data:
                self.cache[card_id] = {
                    'price': price_data.market,
                    'timestamp': datetime.now(),
                    'data': price_data
                }
                return True
            else:
                # API returned None (could be 404 or transient failure)
                # Check if we got stale data instead
                if card_id in self.cache and self.cache[card_id].get('price') is not None:
                    # Keep the existing stale price
                    return True

                # Cache None to track this card was checked
                self.cache[card_id] = {
                    'price': None,
                    'timestamp': datetime.now(),
                    'data': None
                }
                return False

        except Exception as e:
            logger.debug(f"Failed to fetch price for {card_id}: {e}")
            return False
            
    def _get_all_card_ids(self) -> List[str]:
        """Extract all card IDs from feature databases"""
        card_ids: Set[str] = set()
        
        # Check for me1_me2 database first
        me1_me2_path = Path("data/me1_me2_features.pkl")
        if me1_me2_path.exists():
            try:
                with open(me1_me2_path, "rb") as f:
                    data = pickle.load(f)
                    card_ids.update(data.keys())
                    logger.info(f"Found {len(data)} cards in me1_me2_features.pkl")
            except Exception as e:
                logger.error(f"Error reading me1_me2_features.pkl: {e}")
        
        # Also check data/features/ directory as fallback
        features_dir = Path("data/features")
        if features_dir.exists():
            for pkl_file in features_dir.glob("*.pkl"):
                try:
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)
                        card_ids.update(data.keys())
                except Exception as e:
                    logger.error(f"Error reading {pkl_file}: {e}")
                    
        if not card_ids:
            logger.warning("No card IDs found in any feature database")
                
        return sorted(list(card_ids))
        
    def get_price(self, card_id: str) -> float:
        """Get cached price for a card"""
        if card_id in self.cache:
            return self.cache[card_id]['price']
        return None
        
    def get_progress(self) -> Dict:
        """Get current refresh progress"""
        progress_copy = self.progress.copy()
        if self.last_refresh:
            progress_copy['last_refresh'] = self.last_refresh.isoformat()
        return progress_copy
        
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = len(self.cache)
        with_prices = sum(1 for c in self.cache.values() if c['price'] is not None)
        
        return {
            'total_cards': total,
            'cards_with_prices': with_prices,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'is_refreshing': self.is_refreshing
        }
