"""
Price Preloader - Batch loads prices for all known cards
Runs on startup and refreshes periodically
"""
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime, timedelta

from src.api.service import PriceService

logger = logging.getLogger(__name__)


class PricePreloader:
    """
    Preloads prices for all known cards at startup
    Refreshes periodically to keep prices fresh
    """
    
    def __init__(self, price_service: PriceService, refresh_interval_hours: int = 1):
        """
        Initialize price preloader
        
        Args:
            price_service: PriceService instance
            refresh_interval_hours: Auto-refresh interval (default: 1 hour)
        """
        self.service = price_service
        self.refresh_interval = timedelta(hours=refresh_interval_hours)
        self.cache: Dict[str, Dict] = {}  # card_id -> {price, timestamp}
        self.last_refresh = None
        self.refresh_task = None
        self.is_refreshing = False
        self.progress = {"current": 0, "total": 0, "status": "idle"}
        
    async def start(self):
        """Start preloader with initial load"""
        logger.info("Starting price preloader...")
        
        # 1. Load stale prices immediately (Stale-While-Revalidate)
        await self.load_stale_prices()
        
        # 2. Start fresh load in background (non-blocking)
        asyncio.create_task(self.refresh_all_prices())
        
        # 3. Start periodic refresh task
        self.refresh_task = asyncio.create_task(self._periodic_refresh())

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
                        'timestamp': datetime.now(), # UI will see this immediately
                        'data': price_data
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
        """Load prices for all known cards"""
        if self.is_refreshing:
            logger.warning("Refresh already in progress")
            return
            
        self.is_refreshing = True
        self.progress["status"] = "loading"
        
        try:
            # Get all card IDs from feature databases
            card_ids = self._get_all_card_ids()
            total = len(card_ids)
            
            logger.info(f"Starting price refresh for {total} cards...")
            self.progress["total"] = total
            self.progress["current"] = 0
            
            # Fetch prices in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, total, batch_size):
                batch = card_ids[i:i + batch_size]
                
                # Fetch batch concurrently
                tasks = [
                    asyncio.to_thread(self._fetch_and_cache, card_id)
                    for card_id in batch
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                self.progress["current"] = min(i + batch_size, total)
                
                # Small delay between batches to be nice to the API
                await asyncio.sleep(0.5)
                
            self.last_refresh = datetime.now()
            self.progress["status"] = "complete"
            logger.info(f"Price refresh complete. Cached {len(self.cache)} prices.")
            
        except Exception as e:
            logger.error(f"Error refreshing prices: {e}")
            self.progress["status"] = "error"
        finally:
            self.is_refreshing = False
            
    def _fetch_and_cache(self, card_id: str):
        """Fetch and cache a single card's price"""
        try:
            price_data = self.service.get_price(card_id)
            
            if price_data:
                self.cache[card_id] = {
                    'price': price_data.market,
                    'timestamp': datetime.now(),
                    'data': price_data
                }
            else:
                # Cache None to avoid repeated failed fetches
                self.cache[card_id] = {
                    'price': None,
                    'timestamp': datetime.now(),
                    'data': None
                }
        except Exception as e:
            logger.debug(f"Failed to fetch price for {card_id}: {e}")
            
    def _get_all_card_ids(self) -> List[str]:
        """Extract all card IDs from feature databases"""
        card_ids: Set[str] = set()
        features_dir = Path("data/features")
        
        if not features_dir.exists():
            logger.warning("Features directory not found")
            return []
            
        for pkl_file in features_dir.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    card_ids.update(data.keys())
            except Exception as e:
                logger.error(f"Error reading {pkl_file}: {e}")
                
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
