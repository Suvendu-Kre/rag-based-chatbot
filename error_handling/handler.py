import time
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retry(attempts=3, delay=1, exponential_backoff=True):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
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
                    wait_time = delay * (2 ** (attempt - 1) if exponential_backoff else 1)
                    logging.warning(f"Retrying {func.__name__} in {wait_time} seconds...")
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Example usage:
# @retry(attempts=3, delay=1, exponential_backoff=True)
# def my_function():
#     # Function that might fail
#     pass