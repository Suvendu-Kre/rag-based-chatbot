from langchain_core.tools import tool
import ast
import math

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Safely evaluate the expression using ast
        node = ast.parse(expression, mode='eval')

        # Define allowed functions and names
        allowed_names = {
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "nan": math.nan
        }
        allowed_functions = {
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "degrees": math.degrees,
            "radians": math.radians
        }

        # Walk the AST to check for disallowed names and functions
        for node in ast.walk(node):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise NameError(f"Name '{node.id}' is not allowed")
            elif isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions):
                raise NameError(f"Function '{node.func.id}' is not allowed")

        # Evaluate the expression in a safe environment
        code = compile(ast.parse(expression, mode='eval'), filename='<string>', mode='eval')
        result = eval(code, {'__builtins__': {}}, {**allowed_names, **allowed_functions})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_tools():
    return [calculate]