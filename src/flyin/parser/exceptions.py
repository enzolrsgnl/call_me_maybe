class ParsingError(Exception):
    """Indicate that an input file does not follow the expected format."""

    def __init__(self, line_number: int, reason: str) -> None:
        """Initialize the error with the relevant line number and reason."""
        super().__init__(f"Parsing error at line {line_number}: {reason}")
