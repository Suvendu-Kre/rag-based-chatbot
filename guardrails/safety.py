def validate_input(input: str) -> str:
    """Basic input validation."""
    return input.strip() if input and input.strip() else "Error: Empty input."

def validate_output(output: str) -> str:
    """Basic output validation."""
    return output.strip() if output and output.strip() else "Error: Empty response."