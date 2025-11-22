"""
Base classes for Price API integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict


@dataclass
class PriceData:
    """Data model for card price information"""
    
    currency: str = "USD"
    low: Optional[float] = None
    mid: Optional[float] = None
    high: Optional[float] = None
    market: Optional[float] = None
    direct_low: Optional[float] = None  # TCGPlayer direct low
    updated_at: datetime = datetime.now()
    source: str = "unknown"
    url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "currency": self.currency,
            "low": self.low,
            "mid": self.mid,
            "high": self.high,
            "market": self.market,
            "direct_low": self.direct_low,
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "url": self.url,
        }

    @property
    def display_price(self) -> str:
        """Get the best price to display (Market > Mid > Low)"""
        price = self.market or self.mid or self.low
        if price is not None:
            return f"${price:.2f}"
        return "N/A"


class PriceSource(ABC):
    """Abstract base class for price data sources"""

    @abstractmethod
    def get_price(self, card_id: str) -> Optional[PriceData]:
        """
        Fetch price for a specific card
        
        Args:
            card_id: Card ID (e.g., "base1-4")
            
        Returns:
            PriceData object or None if not found
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get source name"""
        pass
