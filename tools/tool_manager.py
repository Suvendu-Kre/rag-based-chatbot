from langchain_core.tools import tool
import ast
import math

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        node = ast.parse(expression, mode='eval')
        # Only allow safe math functions and operators
        safe_list = ['math', 'sqrt', 'pow', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'log', 'log10', 'exp', 'pi']
        for name in node.body.names:
            if name.id not in safe_list:
                return "Error: Invalid expression. Unsafe function or variable used."
        code = compile(node, '<string>', 'eval')
        result = eval(code, {"math": math}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_tools():
    return [calculate]