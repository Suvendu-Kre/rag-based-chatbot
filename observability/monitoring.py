import time
import logging
from functools import wraps
from typing import Callable, Any

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_request() -> Callable:
    """
    Decorator for logging request information.  This version supports async
    handlers and does not execute the handler during import.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                response = await func(*args, **kwargs)
                end_time = time.time()
                logging.info(f"Request to {func.__name__} completed in {end_time - start_time:.4f} seconds.")
                return response
            except Exception as e:
                end_time = time.time()
                logging.error(f"Request to {func.__name__} failed in {end_time - start_time:.4f} seconds: {e}")
                raise
        return wrapper
    return decorator