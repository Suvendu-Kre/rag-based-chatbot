import time
import random
import logging
from typing import Callable, Any

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def retry(func: Callable, attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: The function to retry.
        attempts: The maximum number of attempts.
        delay: The initial delay in seconds.
        exponential_backoff: Whether to use exponential backoff.

    Returns:
        The result of the function if successful, or None if all attempts fail.
    """
    for attempt in range(attempts):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                logging.error(f"All {attempts} attempts failed.")
                raise  # Re-raise the exception after all retries are exhausted
            sleep_time = delay * (2 ** attempt if exponential_backoff else 1) + random.uniform(0, 0.1)
            time.sleep(sleep_time)
    return None