def validate_input(input: str) -> str:
    """
    Validates the input to ensure it is safe and appropriate.
    """
    # Basic input sanitization - prevent script injection
    sanitized_input = input.replace("<", "&lt;").replace(">", "&gt;")
    return sanitized_input.strip() if sanitized_input and sanitized_input.strip() else "Error: Empty input."

def validate_output(output: str) -> str:
    """
    Validates the output to ensure it is non-empty.
    """
    return output.strip() if output and output.strip() else "Error: Empty response."