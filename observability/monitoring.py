import time
import logging
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for logging request information.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = await func(*args, **kwargs)
            status = "success"
            return response
        except Exception as e:
            logging.error(f"Error during request processing: {e}")
            status = "failure"
            raise  # Re-raise the exception after logging
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Request to {func.__name__} completed with status: {status}, duration: {duration:.4f}s")
    return wrapper