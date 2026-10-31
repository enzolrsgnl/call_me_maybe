from .zone import Zone
from typing import Optional
from .exceptions import DroneAlreadyArrivedError, PathNotCompleteError


class Drone:
    """Represent a drone navigating through the zone network."""

    def __init__(
        self,
        drone_id: str,
        current_zone: Optional[Zone],
        path: Optional[list[Zone]] = None,
        in_transit: bool = False,
        has_arrived: bool = False,
        path_index: int = 0,
    ) -> None:
        """Initialize a drone with its starting position and default state."""
        self.drone_id = drone_id
        self.current_zone = current_zone
        self.has_arrived = has_arrived
        self.path = path if path is not None else []
        self.path_index = path_index
        self.in_transit = in_transit

    def advance(self, new_zone: Zone) -> None:
        """Move the drone to a zone, advancing its path index and clearing transit state."""
        if self.has_arrived:
            raise DroneAlreadyArrivedError(self.drone_id)
        self.current_zone = new_zone
        self.path_index += 1
        self.in_transit = False

    def start_transit(self) -> None:
        """Put the drone into transit state, clearing its current zone."""
        self.in_transit = True
        self.current_zone = None

    def arrive(self) -> None:
        """Mark the drone as arrived, provided its path has been fully completed."""
        if self.path_index != len(self.path) - 1:
            raise PathNotCompleteError(self.drone_id)
        self.has_arrived = True
