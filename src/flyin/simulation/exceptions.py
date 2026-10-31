class ScheduleError(Exception):
    """Indicate that the scheduler cannot connect two consecutive zones."""

    def __init__(self, from_zone: str, to_zone: str) -> None:
        """Initialize the error with the unavailable path endpoints."""
        super().__init__(f"No connection found between '{from_zone}' and '{to_zone}'")


class DeadlockError(Exception):
    """Indicate that a drone cannot progress after too many waiting turns."""

    def __init__(self, drone_id: str) -> None:
        """Initialize the error with the blocked drone's identifier."""
        super().__init__(f"Drone '{drone_id}' appears to be stuck (deadlock detected)")
