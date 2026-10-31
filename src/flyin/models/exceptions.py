class ZoneFullError(Exception):
    """Indicate that a drone cannot enter a zone at full capacity."""

    def __init__(self, zone_name: str) -> None:
        """Initialize the error with the full zone's name."""
        super().__init__(f"Zone '{zone_name}' is at full capacity")


class DroneNotInZoneError(Exception):
    """Indicate that a drone is not present in a zone."""

    def __init__(self, zone_name: str, drone_id: str) -> None:
        """Initialize the error with the zone and drone identifiers."""
        super().__init__(f"Drone '{drone_id}' is not in zone '{zone_name}'")


class DroneAlreadyArrivedError(Exception):
    """Indicate that an arrived drone was asked to move."""

    def __init__(self, drone_id: str) -> None:
        """Initialize the error with the arrived drone's identifier."""
        super().__init__(f"Drone '{drone_id}' is already arrived and can't move")


class PathNotCompleteError(Exception):
    """Indicate that a drone was marked arrived before completing its path."""

    def __init__(self, drone_id: str) -> None:
        """Initialize the error with the drone's identifier."""
        super().__init__(f"Drone '{drone_id}' has not finished its complete pass")


class ConnectionFullError(Exception):
    """Indicate that a connection has reached its capacity."""

    def __init__(self, zone_a_name: str, zone_b_name: str) -> None:
        """Initialize the error with the connection's endpoint names."""
        super().__init__(f"Connection between '{zone_a_name}' and '{zone_b_name}' is already full")


class DroneNotInConnectionError(Exception):
    """Indicate that a drone is not present on a connection."""

    def __init__(self, zone_a_name: str, zone_b_name: str, drone_id: str) -> None:
        """Initialize the error with the endpoints and drone identifier."""
        super().__init__(
            f"Drone '{drone_id}' is not in the connection between "
            f"'{zone_a_name}' and '{zone_b_name}'"
        )


class ZoneDuplicateError(Exception):
    """Indicate that a zone is already present in the graph."""

    def __init__(self, zone_name: str) -> None:
        """Initialize the error with the duplicate zone's name."""
        super().__init__(f"The zone '{zone_name}' is already added")


class ConnectionDuplicateError(Exception):
    """Indicate that a connection is already present in the graph."""

    def __init__(self, zone_a_name: str, zone_b_name: str) -> None:
        """Initialize the error with the connection's endpoint names."""
        super().__init__(
            f"The connection between '{zone_a_name}' and '{zone_b_name}' "
            "is already added"
        )
