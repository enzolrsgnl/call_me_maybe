from .zone import Zone
from .exceptions import ConnectionFullError, DroneNotInConnectionError


class Connection:
    """Represent a bidirectional connection between two zones."""

    def __init__(self, zone_a: Zone, zone_b: Zone, max_link_capacity: int = 1) -> None:
        """Initialize a connection with its two endpoints and capacity."""
        self.zone_a = zone_a
        self.zone_b = zone_b
        self.max_link_capacity = max_link_capacity
        self.occupants: set[str] = set()

    def can_accept(self) -> bool:
        """Check whether the connection has capacity for one more drone."""
        return self.max_link_capacity > len(self.occupants)

    def add_drone(self, drone_id: str) -> None:
        """Add a drone to the connection if capacity allows, otherwise raise an error."""
        if self.can_accept():
            self.occupants.add(drone_id)
        else:
            raise ConnectionFullError(self.zone_a.name, self.zone_b.name)

    def remove_drone(self, drone_id: str) -> None:
        """Remove a drone from the connection if it is currently on it."""
        if drone_id not in self.occupants:
            raise DroneNotInConnectionError(self.zone_a.name, self.zone_b.name, drone_id)
        self.occupants.remove(drone_id)
