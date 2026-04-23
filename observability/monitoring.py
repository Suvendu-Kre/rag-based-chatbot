import time
import logging
from functools import wraps
from typing import Callable, Any

def process_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for logging request information.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = await func(*args, **kwargs)
            status_code = 200  # Assume success if no exception
            return response
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            status_code = 500  # Internal Server Error
            raise
        finally:
            duration = time.time() - start_time
            logging.info(
                f"Request to {func.__name__} took {duration:.4f}s and returned status code {status_code}"
            )
    return wrapper