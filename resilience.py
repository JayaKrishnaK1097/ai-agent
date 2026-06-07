"""
Resilience module: retry logic for transient failures when calling the agent.

Uses tenacity for retry-with-exponential-backoff, but only retries on transient
errors (5xx, timeouts, connection issues). Does NOT retry on client errors
(4xx) because those won't recover from being retried.
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import logging

logger = logging.getLogger("ai-agent")

def is_transient_error(exception: BaseException) -> bool:
    """
    Decide whether an exception represents a transient (retryable) failure.
    
    Returns True for: 5xx errors, timeouts, connection errors.
    Returns False for: 4xx client errors, authentication errors.
    """

    error_str = str(exception).lower()

    # 4xx — client's fault, never recovers from retry
    permanent_indivators = [
        "400", "401", "403", "404",
        "bad request", "unauthorized", "forbidden", "not found",
        "invalid api key", "permission denied"
    ]
    if any(indicator in error_str for indicator in permanent_indivators):
        return False
    
    # 5xx + connection issues — worth retrying
    transient_indicators = [
        "500", "502", "503", "504",
        "service unavailable", "internal server error", "gateway timeout",
        "timeout", "connection", "temporarily unavailable",
    ]
    if any(indicator in error_str for indicator in transient_indicators):
        return True

    # Unknown error — default to NOT retrying (safer than retrying everything)
    return False

# Decorator: Retry upto 3 times with exponential backoff starting at 1s, max 8s.
def with_retry(func):
    """
    Wrap a function with retry logic for transient failures.
    
    - Retries up to 3 times (so 4 total attempts max)
    - Waits 1s, 2s, 4s between attempts (exponential)
    - Caps individual wait at 8s
    - Logs each retry attempt
    """
    return retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(is_transient_error), # Retry for any exception.
        before_sleep=lambda retry_state: logger.warning(
            f"event=retry_attempt attempt={retry_state.attempt_number} "
            f"function={func.__name__} "
            f"next_wait_seconds={retry_state.next_action.sleep if retry_state.next_action else None} "
            f"error={str(retry_state.outcome.exception())[:200]}"
        )
    )(func)