"""
Pokemon TCG API Client
Fetches card data and prices from api.pokemontcg.io / dev.pokemontcg.io.

Features:
- Automatic retry with exponential backoff (6 attempts)
- Circuit breaker to avoid hammering dead API
- Error classification (transient vs permanent)
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

from src.api.base import PriceData, PriceSource
from src.api.errors import (
    TransientError,
    PermanentError,
    RateLimitError,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)


class PokemonTCGClient(PriceSource):
    """Client for Pokemon TCG API (pokemontcg.io) with retry and circuit breaker"""

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

        # Timeout configuration
        self.connect_timeout_seconds = float(os.getenv("PRICE_API_CONNECT_TIMEOUT", "5.0"))
        env_timeout = os.getenv("POKEMONTCG_TIMEOUT_SECONDS")
        if env_timeout is None or env_timeout.strip().lower() in {"", "none", "null", "off"}:
            self.default_read_timeout_seconds = None
        else:
            try:
                self.default_read_timeout_seconds = float(env_timeout)
            except Exception:
                self.default_read_timeout_seconds = None

        # Retry configuration
        self.max_retries = int(os.getenv("PRICE_API_MAX_RETRIES", "6"))
        self.retry_min_wait = float(os.getenv("PRICE_API_RETRY_MIN_WAIT", "0.5"))
        self.retry_max_wait = float(os.getenv("PRICE_API_RETRY_MAX_WAIT", "16"))

        # Circuit breaker state
        self._circuit_failure_threshold = int(os.getenv("PRICE_API_CIRCUIT_THRESHOLD", "5"))
        self._circuit_timeout_seconds = int(os.getenv("PRICE_API_CIRCUIT_TIMEOUT", "60"))
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[float] = None
        self._last_success_time: Optional[float] = None

        logger.info(
            f"PokemonTCGClient initialized: base_url={self.base_url}, "
            f"max_retries={self.max_retries}, circuit_threshold={self._circuit_failure_threshold}"
        )

    def get_name(self) -> str:
        return "Pokemon TCG API"

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (should fast-fail)"""
        if self._circuit_open_until is None:
            return False

        if time.time() >= self._circuit_open_until:
            # Circuit timeout expired, move to half-open state
            logger.info("Circuit breaker: timeout expired, moving to half-open state")
            self._circuit_open_until = None
            return False

        return True

    def _record_success(self):
        """Record a successful API call"""
        self._consecutive_failures = 0
        self._last_success_time = time.time()
        if self._circuit_open_until is not None:
            logger.info("Circuit breaker: closed after successful request")
            self._circuit_open_until = None

    def _record_failure(self):
        """Record a failed API call, potentially opening circuit"""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self._circuit_failure_threshold:
            self._circuit_open_until = time.time() + self._circuit_timeout_seconds
            logger.warning(
                f"Circuit breaker: OPEN after {self._consecutive_failures} failures. "
                f"Will retry in {self._circuit_timeout_seconds}s"
            )

    def _fetch_with_retry(self, url: str, timeout) -> dict:
        """
        Fetch URL with automatic retry on transient errors.

        Uses tenacity for exponential backoff with jitter.
        Raises:
            TransientError: On retryable failures (after all retries exhausted)
            PermanentError: On non-retryable failures (404, 400)
            RateLimitError: On rate limiting (429)
            CircuitOpenError: If circuit breaker is open
        """
        # Check circuit breaker first
        if self._is_circuit_open():
            remaining = int(self._circuit_open_until - time.time())
            raise CircuitOpenError(
                f"Circuit breaker open, retry in {remaining}s",
                retry_after=remaining
            )

        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)

            # Handle specific status codes
            if response.status_code == 404:
                # Permanent error - don't retry
                raise PermanentError(f"Card not found (404)")

            if response.status_code == 400:
                raise PermanentError(f"Bad request (400): {response.text[:100]}")

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                self._record_failure()
                raise RateLimitError(
                    f"Rate limit exceeded, retry after {retry_after}s",
                    retry_after=retry_after
                )

            if response.status_code >= 500:
                self._record_failure()
                raise TransientError(f"Server error ({response.status_code})")

            response.raise_for_status()
            self._record_success()
            return response.json()

        except requests.Timeout as e:
            self._record_failure()
            raise TransientError(f"Request timeout: {e}")

        except requests.ConnectionError as e:
            self._record_failure()
            raise TransientError(f"Connection error: {e}")

        except (PermanentError, TransientError, RateLimitError, CircuitOpenError):
            # Re-raise our custom exceptions
            raise

        except requests.RequestException as e:
            self._record_failure()
            raise TransientError(f"Request failed: {e}")

    def get_price(self, card_id: str, timeout_seconds: Optional[float] = None) -> Optional[PriceData]:
        """
        Fetch price for a card with automatic retry.

        Args:
            card_id: Card ID (e.g., "base1-4")
            timeout_seconds: Optional timeout override for this request

        Returns:
            PriceData or None

        Raises:
            PermanentError: If card doesn't exist (404)
            TransientError: If all retries exhausted
            CircuitOpenError: If circuit breaker is open
        """
        url = f"{self.base_url}/cards/{card_id}"
        logger.debug(f"Fetching price for {card_id} from {url}")

        read_timeout = self.default_read_timeout_seconds if timeout_seconds is None else timeout_seconds
        timeout = (self.connect_timeout_seconds, read_timeout)

        # Create retry-decorated function dynamically to use instance config
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(
                initial=self.retry_min_wait,
                max=self.retry_max_wait,
                jitter=2
            ),
            retry=retry_if_exception_type(TransientError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        def fetch_with_retry():
            return self._fetch_with_retry(url, timeout)

        try:
            data = fetch_with_retry()
            return self._parse_price_data(data.get("data", {}))

        except PermanentError as e:
            logger.warning(f"Permanent error for {card_id}: {e}")
            raise

        except (TransientError, RateLimitError, CircuitOpenError) as e:
            logger.error(f"Transient error for {card_id} after retries: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error fetching {card_id}: {e}")
            raise TransientError(f"Unexpected error: {e}")

    def get_price_safe(self, card_id: str, timeout_seconds: Optional[float] = None) -> Optional[PriceData]:
        """
        Fetch price for a card, returning None on any error.

        This is a backward-compatible wrapper that catches all exceptions.
        Prefer get_price() for proper error handling.

        Args:
            card_id: Card ID (e.g., "base1-4")
            timeout_seconds: Optional timeout override

        Returns:
            PriceData or None (never raises)
        """
        try:
            return self.get_price(card_id, timeout_seconds)
        except Exception as e:
            logger.debug(f"get_price_safe caught: {e}")
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

    def get_circuit_status(self) -> dict:
        """Get current circuit breaker status for monitoring"""
        is_open = self._is_circuit_open()
        return {
            "state": "open" if is_open else "closed",
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._circuit_failure_threshold,
            "open_until": self._circuit_open_until,
            "last_success": self._last_success_time,
        }
