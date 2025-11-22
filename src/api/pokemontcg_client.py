"""
Pokemon TCG API Client
Fetches card data and prices from api.pokemontcg.io
"""

import logging
import requests
from typing import Optional, Dict
from datetime import datetime

from src.api.base import PriceSource, PriceData

logger = logging.getLogger(__name__)


class PokemonTCGClient(PriceSource):
    """Client for Pokemon TCG API (pokemontcg.io)"""

    BASE_URL = "https://api.pokemontcg.io/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client
        
        Args:
            api_key: Optional API key (increases rate limit)
        """
        self.api_key = api_key
        self.headers = {"X-Api-Key": api_key} if api_key else {}

    def get_name(self) -> str:
        return "Pokemon TCG API"

    def get_price(self, card_id: str) -> Optional[PriceData]:
        """
        Fetch price for a card
        
        Args:
            card_id: Card ID (e.g., "base1-4")
            
        Returns:
            PriceData or None
        """
        try:
            url = f"{self.BASE_URL}/cards/{card_id}"
            logger.info(f"Fetching price for {card_id} from {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 404:
                logger.warning(f"Card not found: {card_id}")
                return None
                
            if response.status_code == 429:
                logger.warning("Rate limit exceeded for Pokemon TCG API")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            return self._parse_price_data(data.get("data", {}))
            
        except Exception as e:
            logger.error(f"Error fetching price for {card_id}: {e}")
            return None

    def _parse_price_data(self, card_data: Dict) -> Optional[PriceData]:
        """Parse TCGPlayer price data from API response"""
        tcgplayer = card_data.get("tcgplayer", {})
        prices = tcgplayer.get("prices", {})
        url = tcgplayer.get("url")
        
        # Priority: Holofoil > Reverse Holofoil > Normal
        price_type = "holofoil" if "holofoil" in prices else \
                     "reverseHolofoil" if "reverseHolofoil" in prices else \
                     "normal"
                     
        price_info = prices.get(price_type)
        
        if not price_info:
            logger.warning(f"No price info found for {card_data.get('name')}")
            return None
            
        return PriceData(
            currency="USD",
            low=price_info.get("low"),
            mid=price_info.get("mid"),
            high=price_info.get("high"),
            market=price_info.get("market"),
            direct_low=price_info.get("directLow"),
            updated_at=datetime.now(),
            source="tcgplayer",
            url=url
        )
