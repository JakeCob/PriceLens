"""
Smart Caching System
Multi-level cache with Memory (LRU) and Disk (SQLite) layers.
"""

import json
import logging
import sqlite3
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from src.api.base import PriceData

logger = logging.getLogger(__name__)

class SmartCache:
    """
    Two-level cache system:
    1. Memory Cache (Fastest, Volatile)
    2. SQLite Cache (Fast, Persistent)
    """

    def __init__(self, db_path: str = "data/cache/cache.db", ttl_seconds: int = 3600):
        """
        Initialize SmartCache
        
        Args:
            db_path: Path to SQLite database
            ttl_seconds: Time to live for cache entries
        """
        self.db_path = Path(db_path)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.memory_cache: Dict[str, Any] = {}
        
        self._init_db()
        logger.info(f"SmartCache initialized (TTL={ttl_seconds}s)")

    def _init_db(self) -> None:
        """Initialize SQLite database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # 1. Check Memory
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Memory cache hit for {key}")
                return value
            else:
                del self.memory_cache[key]

        # 2. Check Disk (SQLite)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, created_at FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value_blob, created_at_str = row
                    created_at = datetime.fromisoformat(created_at_str)
                    
                    if datetime.now() - created_at < self.ttl:
                        value = pickle.loads(value_blob)
                        # Promote to memory
                        self.memory_cache[key] = (value, datetime.now())
                        logger.debug(f"Disk cache hit for {key}")
                        return value
                    else:
                        # Expired
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        now = datetime.now()
        
        # 1. Update Memory
        self.memory_cache[key] = (value, now)
        
        # 2. Update Disk
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                    (key, pickle.dumps(value), now.isoformat())
                )
        except Exception as e:
            logger.error(f"Cache write error: {e}")

    def clear_expired(self) -> None:
        """Remove expired entries from DB"""
        cutoff = (datetime.now() - self.ttl).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
                logger.info("Cleared expired cache entries")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
