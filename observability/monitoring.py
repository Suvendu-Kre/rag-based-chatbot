import logging
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for logging request details and any exceptions that occur.
    This version is FastAPI-safe and supports async handlers.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            logging.info(f"Request received for: {func.__name__}")
            result = await func(*args, **kwargs)
            logging.info(f"Request completed for: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise  # Re-raise the exception after logging
    return wrapper