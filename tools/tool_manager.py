from langchain_core.tools import tool
import ast
import math

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        node = ast.parse(expression, mode='eval')
        # Only allow safe math functions
        safe_list = ['math', 'sqrt', 'pow', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'log', 'log10', 'exp']
        for name, obj in math.__dict__.items():
            if name in safe_list:
                locals()[name] = obj
        code = compile(node, '<string>', 'eval')
        result = eval(code, {'__builtins__': None}, locals())
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_tools():
    return [calculate]