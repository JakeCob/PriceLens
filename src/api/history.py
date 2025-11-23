"""
Price history storage and trend computation.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.api.base import PriceData


class PriceHistoryRepository:
    """SQLite-backed repository for card price history."""

    def __init__(self, db_path: str = "data/prices/price_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    source TEXT,
                    currency TEXT,
                    low REAL,
                    mid REAL,
                    high REAL,
                    market REAL,
                    direct_low REAL,
                    fetched_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_history_card_time ON price_history(card_id, fetched_at DESC);"
            )

    def record(self, card_id: str, price: PriceData) -> None:
        """Persist a new price snapshot for a card."""
        fetched_at = (price.updated_at or datetime.utcnow()).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO price_history
                (card_id, source, currency, low, mid, high, market, direct_low, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    card_id,
                    price.source,
                    price.currency,
                    price.low,
                    price.mid,
                    price.high,
                    price.market,
                    price.direct_low,
                    fetched_at,
                ),
            )

    def get_recent(self, card_id: str, limit: int = 2) -> List[PriceData]:
        """Return the most recent price snapshots for a card."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT source, currency, low, mid, high, market, direct_low, fetched_at
                FROM price_history
                WHERE card_id = ?
                ORDER BY datetime(fetched_at) DESC
                LIMIT ?;
                """,
                (card_id, limit),
            ).fetchall()

        prices: List[PriceData] = []
        for row in rows:
            (
                source,
                currency,
                low,
                mid,
                high,
                market,
                direct_low,
                fetched_at,
            ) = row

            prices.append(
                PriceData(
                    currency=currency or "USD",
                    low=low,
                    mid=mid,
                    high=high,
                    market=market,
                    direct_low=direct_low,
                    updated_at=datetime.fromisoformat(fetched_at),
                    source=source or "unknown",
                )
            )
        return prices

    def get_trend(self, card_id: str) -> Tuple[Optional[float], Optional[PriceData], Optional[PriceData]]:
        """
        Compute percentage change between the latest two price entries.

        Returns:
            (change_percent, current, previous)
            change_percent is None if insufficient data.
        """
        recent = self.get_recent(card_id, limit=2)
        if len(recent) < 2:
            return None, recent[0] if recent else None, None

        current, previous = recent[0], recent[1]
        current_value = current.market or current.mid or current.low
        previous_value = previous.market or previous.mid or previous.low

        if previous_value in (None, 0) or current_value is None:
            return None, current, previous

        change_percent = ((current_value - previous_value) / previous_value) * 100
        return change_percent, current, previous


def price_to_dict_with_trend(
    card_id: str, price: Optional[PriceData], trend_percent: Optional[float]
) -> dict:
    """Helper to package price + trend for API responses."""
    if not price:
        return {}
    data = price.to_dict()
    data["card_id"] = card_id
    data["trend_percent"] = trend_percent
    return data
