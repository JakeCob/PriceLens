"""
Background Price Queue for Live Detection
Fetches prices asynchronously to avoid blocking live detection frames
"""
import asyncio
import logging
from collections import deque
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.api.service import PriceService

logger = logging.getLogger(__name__)


class PriceQueue:
    """
    Background worker that fetches prices without blocking detection
    """
    
    def __init__(self, price_service: PriceService, cache_ttl: int = 300):
        """
        Initialize price queue
        
        Args:
            price_service: PriceService instance
            cache_ttl: How long to cache prices (seconds)
        """
        self.service = price_service
        self.cache_ttl = timedelta(seconds=cache_ttl)
        self.queue = deque()
        self.cache: Dict[str, Dict] = {}  # card_id -> {price, timestamp}
        self.task = None
        self.running = False
        
    async def start(self):
        """Start background worker"""
        if self.running:
            return
            
        self.running = True
        self.task = asyncio.create_task(self._worker())
        logger.info("PriceQueue worker started")
        
    async def stop(self):
        """Stop background worker"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("PriceQueue worker stopped")
            
    async def _worker(self):
        """Process price requests in background"""
        while self.running:
            try:
                if self.queue:
                    card_id = self.queue.popleft()
                    
                    # Skip if already cached and not expired
                    if self._is_cached(card_id):
                        continue
                    
                    # Fetch price synchronously (runs in event loop)
                    # TODO: Make price_service.get_price async
                    price_data = await asyncio.to_thread(
                        self.service.get_price, card_id
                    )
                    
                    if price_data:
                        self.cache[card_id] = {
                            'price': price_data.market,
                            'timestamp': datetime.now(),
                            'data': price_data
                        }
                        logger.debug(f"Cached price for {card_id}: ${price_data.market}")
                    else:
                        # Cache None to avoid repeated failed fetches
                        self.cache[card_id] = {
                            'price': None,
                            'timestamp': datetime.now(),
                            'data': None
                        }
                        
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                
            except Exception as e:
                logger.error(f"Price queue worker error: {e}")
                await asyncio.sleep(1)  # Longer delay on error
                
    def _is_cached(self, card_id: str) -> bool:
        """Check if price is cached and not expired"""
        if card_id not in self.cache:
            return False
            
        cached = self.cache[card_id]
        age = datetime.now() - cached['timestamp']
        return age < self.cache_ttl
        
    def request_price(self, card_id: str):
        """Add card to price fetch queue"""
        if not self._is_cached(card_id) and card_id not in self.queue:
            self.queue.append(card_id)
            logger.debug(f"Queued price request for {card_id}")
            
    def get_cached_price(self, card_id: str) -> Optional[float]:
        """Get price from cache if available"""
        if self._is_cached(card_id):
            return self.cache[card_id]['price']
        return None
        
    def get_queue_size(self) -> int:
        """Get number of pending requests"""
        return len(self.queue)
        
    def clear_cache(self):
        """Clear all cached prices"""
        self.cache.clear()
        logger.info("Price cache cleared")
