"""
Custom Price Source
Reads prices from a local JSON file. Useful for custom sets or overriding API prices.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.api.base import PriceData, PriceSource

logger = logging.getLogger(__name__)

class CustomPriceSource(PriceSource):
    """
    Price source that reads from a local JSON file.
    File format:
    {
        "card_id": {
            "price": 10.50,
            "currency": "USD"
        }
    }
    """

    def __init__(self, file_path: str = "data/custom_prices.json"):
        self.file_path = Path(file_path)
        self.prices: Dict = {}
        self._load_prices()

    def get_name(self) -> str:
        return "Custom Local Prices"

    def _load_prices(self) -> None:
        """Load prices from JSON file"""
        if not self.file_path.exists():
            logger.warning(f"Custom price file not found at {self.file_path}")
            return

        try:
            with open(self.file_path, "r") as f:
                self.prices = json.load(f)
            logger.info(f"Loaded {len(self.prices)} custom prices")
        except Exception as e:
            logger.error(f"Failed to load custom prices: {e}")

    def get_price(self, card_id: str) -> Optional[PriceData]:
        """Get price for card_id"""
        # Reload occasionally? For now, just use memory cache from init
        # In a real app, we might want to watch the file for changes
        
        if card_id not in self.prices:
            return None

        data = self.prices[card_id]
        
        # Handle simple float or dict
        if isinstance(data, (int, float)):
            price_val = float(data)
        else:
            price_val = float(data.get("price", 0.0))

        return PriceData(
            currency="USD",
            low=None,
            mid=price_val,
            high=None,
            market=price_val,
            direct_low=None,
            updated_at=datetime.now(),
            source="custom",
            url=None
        )

    def set_price(self, card_id: str, price: float) -> None:
        """Update a price"""
        self.prices[card_id] = price
        self._save_prices()

    def _save_prices(self) -> None:
        """Save to disk"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(self.prices, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom prices: {e}")
