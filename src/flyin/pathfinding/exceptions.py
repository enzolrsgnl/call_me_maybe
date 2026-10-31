class PathNotFoundError(Exception):
    """Indicate that no valid path exists between two zones."""

    def __init__(self, start_name: str, end_name: str) -> None:
        """Initialize the error with the start and end zone names."""
        super().__init__(f"No path found between '{start_name}' and '{end_name}'")
