"""
Smart Caching System
Multi-level cache with Memory (LRU) and Disk (SQLite) layers.

Features:
- Two-level caching (memory + SQLite)
- Stale-while-revalidate support
- Failure tracking with short TTL to avoid hammering failed requests
"""

import json
import logging
import os
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

    Also tracks recent failures to avoid repeatedly hammering failing requests.
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

        # Failure tracking (in-memory only, short TTL)
        self.failure_cache: Dict[str, Dict] = {}
        self.failure_ttl_seconds = int(os.getenv("PRICE_FAILURE_CACHE_TTL", "300"))

        self._init_db()
        logger.info(f"SmartCache initialized (TTL={ttl_seconds}s, failure_ttl={self.failure_ttl_seconds}s)")

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

    def get_stale(self, key: str) -> Optional[Any]:
        """
        Get value from cache even if expired (for Stale-While-Revalidate)
        """
        # 1. Check Memory
        if key in self.memory_cache:
            return self.memory_cache[key][0]

        # 2. Check Disk (SQLite)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value = pickle.loads(row[0])
                    # Promote to memory (with old timestamp)
                    self.memory_cache[key] = (value, datetime.min) 
                    return value
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

    # --- Failure Tracking ---

    def set_failure(self, key: str, error_type: str, ttl_seconds: Optional[int] = None) -> None:
        """
        Cache a failure with short TTL for retry prevention.

        Failures are stored in memory only (not persisted to disk).
        This prevents hammering a failing API while allowing retries after TTL.

        Args:
            key: Cache key (typically card_id)
            error_type: Description of the error (for logging/debugging)
            ttl_seconds: Override failure TTL (defaults to PRICE_FAILURE_CACHE_TTL)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.failure_ttl_seconds
        self.failure_cache[key] = {
            "error": error_type,
            "timestamp": datetime.now(),
            "ttl": timedelta(seconds=ttl),
        }
        logger.debug(f"Cached failure for {key}: {error_type} (TTL={ttl}s)")

    def is_recently_failed(self, key: str) -> bool:
        """
        Check if a key failed recently (within failure TTL).

        Use this to avoid repeated API calls for known-failing keys.

        Args:
            key: Cache key to check

        Returns:
            True if key failed recently and shouldn't be retried yet
        """
        if key not in self.failure_cache:
            return False

        entry = self.failure_cache[key]
        age = datetime.now() - entry["timestamp"]

        if age < entry["ttl"]:
            logger.debug(f"Key {key} recently failed ({age.total_seconds():.0f}s ago)")
            return True

        # Failure TTL expired, remove entry
        del self.failure_cache[key]
        return False

    def clear_failure(self, key: str) -> None:
        """
        Clear a failure entry (e.g., after successful fetch).

        Args:
            key: Cache key to clear
        """
        if key in self.failure_cache:
            del self.failure_cache[key]
            logger.debug(f"Cleared failure for {key}")

    def get_failure_info(self, key: str) -> Optional[Dict]:
        """
        Get failure information for a key.

        Args:
            key: Cache key

        Returns:
            Dict with error info or None if not failed
        """
        if key not in self.failure_cache:
            return None

        entry = self.failure_cache[key]
        age = datetime.now() - entry["timestamp"]

        if age >= entry["ttl"]:
            del self.failure_cache[key]
            return None

        return {
            "error": entry["error"],
            "age_seconds": age.total_seconds(),
            "ttl_remaining": (entry["ttl"] - age).total_seconds(),
        }

    def get_failure_stats(self) -> Dict:
        """
        Get statistics about current failure cache.

        Returns:
            Dict with failure cache stats
        """
        # Clean up expired entries first
        now = datetime.now()
        expired_keys = [
            k for k, v in self.failure_cache.items()
            if now - v["timestamp"] >= v["ttl"]
        ]
        for k in expired_keys:
            del self.failure_cache[k]

        return {
            "total_failures": len(self.failure_cache),
            "keys": list(self.failure_cache.keys())[:10],  # First 10 for debugging
        }
