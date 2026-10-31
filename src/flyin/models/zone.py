from enum import Enum
from typing import Optional
from .exceptions import ZoneFullError, DroneNotInZoneError


class ZoneType(Enum):
    """Define the movement rules associated with a zone."""

    normal = "NORMAL"
    blocked = "BLOCKED"
    restricted = "RESTRICTED"
    priority = "PRIORITY"

    @property
    def movement_cost(self) -> int:
        """Return the movement cost associated with this zone type."""
        if self == ZoneType.restricted:
            return 2
        elif self == ZoneType.blocked:
            return -1
        return 1

    @property
    def is_authorized(self) -> bool:
        """Return whether this zone type may be traversed."""
        if self == ZoneType.blocked:
            return False
        return True


class Zone:
    """Represent a zone with coordinates, type, capacity, and occupants."""

    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        zone_type: ZoneType,
        max_capacity: int,
        is_unlimited: bool,
        color: Optional[str] = None,
    ) -> None:
        """Initialize a zone with its properties and no occupants."""
        self.name = name
        self.x = x
        self.y = y
        self.zone_type = zone_type
        self.color = color
        self.max_capacity = max_capacity
        self.occupants: set[str] = set()
        self.is_unlimited = is_unlimited

    def can_accept(self) -> bool:
        """Return whether the zone can accept another drone."""
        if self.is_unlimited:
            return True
        return len(self.occupants) < self.max_capacity

    def add_drone(self, drone_id: str) -> None:
        """Add a drone to the zone if capacity allows."""
        if self.can_accept():
            self.occupants.add(drone_id)
        else:
            raise ZoneFullError(self.name)

    def remove_drone(self, drone_id: str) -> None:
        """Remove a drone from the zone if it is currently present."""
        if drone_id not in self.occupants:
            raise DroneNotInZoneError(drone_id, self.name)
        self.occupants.remove(drone_id)
