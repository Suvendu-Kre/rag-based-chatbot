import time
import logging
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for logging request information and timing.  This version supports async functions.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            logging.info(f"Request started: {func.__name__} with args={args} kwargs={kwargs}")
            result = await func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"Request completed: {func.__name__} in {end_time - start_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            logging.error(f"Request failed: {func.__name__} in {end_time - start_time:.4f} seconds with error: {e}")
            raise  # Re-raise the exception
    return wrapper