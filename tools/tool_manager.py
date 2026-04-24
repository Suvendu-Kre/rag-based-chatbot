from langchain_core.tools import tool

@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    """
    Returns the current date formatted according to the specified format.
    The default format is YYYY-MM-DD.
    """
    import datetime
    return datetime.datetime.now().strftime(format)

def get_tools():
    return [get_current_date]