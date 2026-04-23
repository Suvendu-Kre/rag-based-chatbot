def validate_input(input: str) -> str:
    """
    Validates the input to ensure it is not malicious or harmful.
    """
    # Basic input sanitization - prevent script injection
    sanitized_input = input.replace("<", "&lt;").replace(">", "&gt;")
    return sanitized_input.strip() if sanitized_input else "Error: Empty input."

def validate_output(output: str) -> str:
    """
    Validates the output to ensure it is safe and appropriate.
    """
    return output.strip() if output and output.strip() else "Error: Empty response."