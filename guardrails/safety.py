def validate_input(input_text: str) -> str:
    """
    Validates the input text.
    """
    if not input_text or not input_text.strip():
        return "Error: Input cannot be empty."
    return input_text.strip()

def validate_output(output: str) -> str:
    """
    Validates the output.
    """
    return output.strip() if output and output.strip() else "Error: Empty response."