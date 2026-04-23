import time
import random
import logging
from typing import Callable, Any

def retry(func: Callable[..., Any], attempts: int = 3, delay: int = 1, exponential_backoff: bool = True) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        attempt = 0
        while attempt < attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt == attempts:
                    logging.error(f"Function {func.__name__} failed after {attempts} attempts: {e}")
                    raise
                sleep_time = delay * (2 ** (attempt - 1) if exponential_backoff else 1) + random.random()
                logging.warning(f"Retrying {func.__name__} in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
    return wrapper