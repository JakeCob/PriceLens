"""
TCGPlayer Price Scraper
Fallback price source when Pokemon TCG API is unavailable.

Features:
- Extracts market price from TCGPlayer product pages
- Rate-limited to be polite (default: 2s between requests)
- Configurable via environment variables

WARNING: Web scraping may violate TCGPlayer's Terms of Service.
Use responsibly and only as a fallback when API is unavailable.
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.api.base import PriceData

logger = logging.getLogger(__name__)


class TCGPlayerScraper:
    """
    Scrapes price data from TCGPlayer as a fallback when API fails.

    This is a last-resort fallback - prefer the Pokemon TCG API whenever possible.
    """

    def __init__(self):
        """Initialize scraper with rate limiting"""
        self.enabled = os.getenv("TCGPLAYER_SCRAPE_ENABLED", "true").lower() in ("true", "1", "yes")
        self.delay_seconds = float(os.getenv("TCGPLAYER_SCRAPE_DELAY", "2.0"))
        self._last_request_time: Optional[float] = None

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

        if not self.enabled:
            logger.info("TCGPlayer scraper is disabled")
        else:
            logger.info(f"TCGPlayer scraper initialized (delay={self.delay_seconds}s)")

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.delay_seconds:
                sleep_time = self.delay_seconds - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def get_price_from_url(self, url: str) -> Optional[PriceData]:
        """
        Scrape price from a TCGPlayer URL.

        Args:
            url: TCGPlayer product URL (e.g., https://www.tcgplayer.com/product/...)

        Returns:
            PriceData or None if scraping fails
        """
        if not self.enabled:
            logger.debug("Scraper disabled, skipping")
            return None

        if not url or "tcgplayer.com" not in url.lower():
            logger.warning(f"Invalid TCGPlayer URL: {url}")
            return None

        # Handle prices.pokemontcg.io redirect URLs
        if "prices.pokemontcg.io" in url:
            url = self._resolve_redirect(url)
            if not url:
                return None

        self._rate_limit()

        try:
            logger.debug(f"Scraping TCGPlayer: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            return self._parse_price_page(response.text, url)

        except requests.Timeout:
            logger.warning(f"Timeout scraping {url}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Request error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None

    def _resolve_redirect(self, url: str) -> Optional[str]:
        """Resolve prices.pokemontcg.io redirect to actual TCGPlayer URL"""
        try:
            self._rate_limit()
            response = self.session.head(url, allow_redirects=True, timeout=10)
            final_url = response.url
            if "tcgplayer.com" in final_url:
                logger.debug(f"Resolved redirect: {url} -> {final_url}")
                return final_url
            logger.warning(f"Redirect did not lead to TCGPlayer: {final_url}")
            return None
        except Exception as e:
            logger.warning(f"Failed to resolve redirect {url}: {e}")
            return None

    def _parse_price_page(self, html: str, url: str) -> Optional[PriceData]:
        """
        Parse TCGPlayer product page HTML to extract price.

        TCGPlayer's HTML structure changes frequently, so this may need updates.
        """
        soup = BeautifulSoup(html, "lxml")

        # Try multiple selectors (TCGPlayer changes their HTML frequently)
        market_price = self._extract_market_price(soup)
        low_price = self._extract_low_price(soup)

        if market_price is None and low_price is None:
            logger.warning(f"Could not extract any price from {url}")
            return None

        return PriceData(
            currency="USD",
            low=low_price,
            mid=None,
            high=None,
            market=market_price or low_price,  # Use low if no market
            direct_low=None,
            updated_at=datetime.now(),
            source="tcgplayer_scrape",
            url=url,
        )

    def _extract_market_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract market price from page"""
        # Try various selectors that TCGPlayer has used

        # Method 1: Look for "Market Price" label
        market_label = soup.find(string=re.compile(r"Market\s*Price", re.I))
        if market_label:
            parent = market_label.find_parent()
            if parent:
                price_text = parent.get_text()
                price = self._parse_price_text(price_text)
                if price:
                    return price

        # Method 2: Look for price in specific class patterns
        price_selectors = [
            ".price-point__data",
            ".product-details__market-price",
            '[data-testid="price"]',
            ".spotlight__price",
        ]
        for selector in price_selectors:
            elem = soup.select_one(selector)
            if elem:
                price = self._parse_price_text(elem.get_text())
                if price:
                    return price

        # Method 3: Look for any element containing a dollar amount after "Market"
        for text in soup.stripped_strings:
            if "market" in text.lower():
                price = self._parse_price_text(text)
                if price:
                    return price

        return None

    def _extract_low_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract lowest listed price from page"""
        # Look for "Low" or "From" price indicators
        low_patterns = [
            r"Low\s*Price",
            r"From\s*\$",
            r"Starting\s*at",
        ]

        for pattern in low_patterns:
            match = soup.find(string=re.compile(pattern, re.I))
            if match:
                parent = match.find_parent()
                if parent:
                    price = self._parse_price_text(parent.get_text())
                    if price:
                        return price

        return None

    def _parse_price_text(self, text: str) -> Optional[float]:
        """Extract numeric price from text containing dollar amounts"""
        if not text:
            return None

        # Find dollar amounts like $12.34 or $1,234.56
        match = re.search(r"\$\s*([\d,]+\.?\d*)", text)
        if match:
            try:
                price_str = match.group(1).replace(",", "")
                price = float(price_str)
                if 0.01 <= price <= 100000:  # Sanity check
                    return price
            except ValueError:
                pass

        return None

    def get_price(self, card_id: str) -> Optional[PriceData]:
        """
        Get price by card_id (constructs TCGPlayer search URL).

        Note: This is less reliable than using a direct URL.
        Prefer get_price_from_url() when you have the URL.
        """
        if not self.enabled:
            return None

        # Construct search URL
        # card_id format is typically "set-number" like "base1-4"
        parts = card_id.rsplit("-", 1)
        if len(parts) != 2:
            logger.warning(f"Cannot parse card_id for search: {card_id}")
            return None

        set_id, number = parts

        # This is a heuristic - may not work for all cards
        search_url = f"https://www.tcgplayer.com/search/pokemon/product?q={set_id}+{number}"

        self._rate_limit()

        try:
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()

            # Try to find product link in search results
            soup = BeautifulSoup(response.text, "lxml")
            product_link = soup.select_one('a[href*="/product/"]')

            if product_link:
                product_url = product_link.get("href")
                if product_url and not product_url.startswith("http"):
                    product_url = f"https://www.tcgplayer.com{product_url}"
                return self.get_price_from_url(product_url)

            logger.warning(f"No product found in search for {card_id}")
            return None

        except Exception as e:
            logger.warning(f"Search failed for {card_id}: {e}")
            return None
