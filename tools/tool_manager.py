from langchain_core.tools import tool
import ast
import math

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        node = ast.parse(expression, mode='eval')

        # Whitelist allowed functions and names
        safe_list = ['math', 'sqrt', 'pow', 'sin', 'cos', 'tan', 'pi', 'e', 'log', 'log10', 'exp', 'ceil', 'floor']
        safe_dict = {k: getattr(math, k) for k in safe_list if hasattr(math, k)}
        safe_dict['__builtins__'] = None

        result = eval(compile(node, '<string>', 'eval'), safe_dict, None)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_tools():
    return [calculate]