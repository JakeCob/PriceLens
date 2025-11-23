"""
Pokemon TCG API Client
Fetches card data and prices from api.pokemontcg.io / dev.pokemontcg.io.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

from src.api.base import PriceData, PriceSource

logger = logging.getLogger(__name__)


class PokemonTCGClient(PriceSource):
    """Client for Pokemon TCG API (pokemontcg.io)"""

    PROD_BASE_URL = "https://api.pokemontcg.io/v2"
    DEV_BASE_URL = "https://dev.pokemontcg.io/v2"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize client
        
        Args:
            api_key: Optional API key (increases rate limit)
            base_url: Override base URL (defaults to dev if key present, else prod)
        """
        load_dotenv()

        # Support both legacy and dev-prefixed env var names
        env_key = os.getenv("POKEMONTCG_API_KEY") or os.getenv("DEV_POKEMONTCG_IO_API_KEY")
        env_base = os.getenv("POKEMONTCG_BASE_URL") or os.getenv("DEV_POKEMONTCG_IO_BASE_URL")

        self.api_key = api_key or env_key
        self.base_url = (
            base_url
            or env_base
            or (self.DEV_BASE_URL if self.api_key else self.PROD_BASE_URL)
        )
        self.headers = {"X-Api-Key": self.api_key} if self.api_key else {}

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
            url = f"{self.base_url}/cards/{card_id}"
            logger.info(f"Fetching price for %s from %s", card_id, url)
            
            response = requests.get(url, headers=self.headers, timeout=120)
            
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
