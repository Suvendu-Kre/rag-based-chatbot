import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retry(func, attempts=3, delay=1, backoff=2):
    """Retry a function with exponential backoff."""
    for attempt in range(attempts):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == attempts:
                raise
            sleep_time = delay * (backoff ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)

def circuit_breaker(func, failure_threshold=3, recovery_timeout=30):
    """Circuit breaker pattern."""
    state = "CLOSED"
    failure_count = 0
    last_failure_time = 0

    def wrapper(*args, **kwargs):
        nonlocal state, failure_count, last_failure_time

        if state == "OPEN":
            if time.time() - last_failure_time < recovery_timeout:
                raise Exception("Circuit breaker is OPEN.")
            else:
                state = "HALF_OPEN"

        try:
            result = func(*args, **kwargs)
            failure_count = 0  # Reset failure count on success
            state = "CLOSED"
            return result
        except Exception as e:
            failure_count += 1
            last_failure_time = time.time()
            if failure_count >= failure_threshold:
                state = "OPEN"
                logging.error("Circuit breaker OPEN.")
            raise

    return wrapper