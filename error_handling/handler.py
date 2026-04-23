import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retry(func, attempts=3, delay=1, exponential_backoff=True):
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
                sleep_time = delay * (2 ** (attempt - 1) if exponential_backoff else 1) + random.uniform(0, 1)
                logging.warning(f"Attempt {attempt} failed. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
    return wrapper

# Placeholder for circuit breaker implementation (can be added later)