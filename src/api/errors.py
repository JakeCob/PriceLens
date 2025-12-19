"""
Price API Error Types
Distinguishes between transient (retryable) and permanent (non-retryable) errors.
"""


class PriceAPIError(Exception):
    """Base exception for price API errors"""
    pass


class TransientError(PriceAPIError):
    """
    Retryable error - temporary failures that may succeed on retry.

    Examples:
        - Network timeout
        - Connection reset
        - Server errors (5xx)
        - Rate limiting (429)
    """
    pass


class PermanentError(PriceAPIError):
    """
    Non-retryable error - failures that won't succeed on retry.

    Examples:
        - Card not found (404)
        - Bad request (400)
        - Invalid card ID format
    """
    pass


class RateLimitError(TransientError):
    """
    Rate limited by API - includes retry_after hint if available.

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header)
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitOpenError(TransientError):
    """
    Circuit breaker is open - fast-failing to avoid hammering dead API.

    Attributes:
        retry_after: Seconds until circuit will attempt to close
    """

    def __init__(self, message: str = "Circuit breaker open", retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after
