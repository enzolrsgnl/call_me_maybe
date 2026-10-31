from typing import Optional

from .exceptions import ZoneDuplicateError, ConnectionDuplicateError
from .zone import Zone
from .connection import Connection


class Graph:
    """Represent the queryable graph of zones and connections."""

    def __init__(self) -> None:
        """Initialize empty zones, connections, and optional endpoints."""
        self.zones: dict[str, Zone] = {}
        self.connections: list[Connection] = []
        self.start_zone: Optional[Zone] = None
        self.end_zone: Optional[Zone] = None

    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the graph unless its name is already present."""
        if zone.name in self.zones:
            raise ZoneDuplicateError(zone.name)
        self.zones[zone.name] = zone

    def add_connection(self, connection: Connection) -> None:
        """Add a connection unless an equivalent bidirectional link exists."""
        for existing in self.connections:
            if (
                (
                    existing.zone_a.name == connection.zone_a.name
                    and existing.zone_b.name == connection.zone_b.name
                )
                or (
                    existing.zone_a.name == connection.zone_b.name
                    and existing.zone_b.name == connection.zone_a.name
                )
            ):
                raise ConnectionDuplicateError(
                    connection.zone_a.name,
                    connection.zone_b.name,
                )
        self.connections.append(connection)

    def get_neighbors(self, zone: Zone) -> list[tuple[Zone, Connection, int]]:
        """Return each authorized neighboring zone, link, and movement cost."""
        neighbors: list[tuple[Zone, Connection, int]] = []
        for connection in self.connections:
            if connection.zone_a == zone:
                neighbor = connection.zone_b
                if not neighbor.zone_type.is_authorized:
                    continue
                else:
                    cost = neighbor.zone_type.movement_cost
                    neighbors.append((neighbor, connection, cost))
            elif connection.zone_b == zone:
                neighbor = connection.zone_a
                if not neighbor.zone_type.is_authorized:
                    continue
                else:
                    cost = neighbor.zone_type.movement_cost
                    neighbors.append((neighbor, connection, cost))
            else:
                continue
        return neighbors
