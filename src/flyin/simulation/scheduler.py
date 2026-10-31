from ..models import Graph, Drone, Zone, Connection
from ..pathfinding import Pathfinder
from typing import Union, cast
from .exceptions import ScheduleError, DeadlockError


class Scheduler:
    """Computes and manages the movement planning for a fleet of drones."""
    def __init__(self, graph: Graph, drones: list[Drone]) -> None:
        """Initialize the scheduler with the graph and the fleet of drones to route."""
        self.graph = graph
        self.drones = drones
        self.zone_reservations: dict[tuple[int, str], set[str]] = {}
        self.connection_reservations: dict[tuple[int, str, str], set[str]] = {}
        self.pathfinder = Pathfinder()
        self.drone_schedules: dict[str, dict[int, Union[Zone, Connection]]] = {}

    def path_assigner(self) -> None:
        """Compute and assign the individual optimal path for each drone."""
        for drone in self.drones:
            path, _ = self.pathfinder.find_path(
                self.graph,
                cast(Zone, drone.current_zone),
                cast(Zone, self.graph.end_zone),
            )
            drone.path = path

    def _is_zone_available(self, zone: Zone, turn: int) -> bool:
        """Return whether a zone has capacity at a given turn."""
        if zone.is_unlimited:
            return True
        reserved = self.zone_reservations.get((turn, zone.name), set())
        return len(reserved) < zone.max_capacity

    def _reserve_zone(self, zone: Zone, turn: int, drone_id: str) -> None:
        """Reserve a slot for a drone in the given zone at the given turn."""
        reserved = self.zone_reservations.setdefault((turn, zone.name), set())
        reserved.add(drone_id)

    def _is_connection_available(self, connection: Connection, turn: int) -> bool:
        """Return whether a connection has capacity at a given turn."""
        reserved = self.connection_reservations.get(
            (turn, connection.zone_a.name, connection.zone_b.name),
            set(),
        )
        return len(reserved) < connection.max_link_capacity

    def _reserve_connection(self, connection: Connection, turn: int, drone_id: str) -> None:
        """Reserve a slot for a drone in the given connection at the given turn."""
        reserved = self.connection_reservations.setdefault(
            (turn, connection.zone_a.name, connection.zone_b.name),
            set(),
        )
        reserved.add(drone_id)

    def _build_schedule(self) -> None:
        """Compute a turn-by-turn schedule for each drone, resolving conflicts."""
        for drone in self.drones:
            current_turn = 0
            current_zone = cast(Zone, drone.current_zone)
            self.drone_schedules[drone.drone_id] = {current_turn: current_zone}
            path_index = 1
            waiting_turns = 0

            while path_index < len(drone.path):
                next_zone = drone.path[path_index]

                connection = None
                for neighbor_zone, candidate_connection, _ in self.graph.get_neighbors(
                    current_zone
                ):
                    if neighbor_zone.name == next_zone.name:
                        connection = candidate_connection
                        break

                if connection is None:
                    raise ScheduleError(current_zone.name, next_zone.name)

                if next_zone.zone_type.movement_cost == 2:
                    arrival_turn = current_turn + 2
                else:
                    arrival_turn = current_turn + 1

                zone_ok = self._is_zone_available(next_zone, arrival_turn)
                connection_ok = self._is_connection_available(connection, current_turn + 1)

                if zone_ok and connection_ok:
                    self._reserve_connection(connection, current_turn + 1, drone.drone_id)
                    self._reserve_zone(next_zone, arrival_turn, drone.drone_id)

                    if arrival_turn == current_turn + 2:
                        self.drone_schedules[drone.drone_id][current_turn + 1] = connection

                    current_turn = arrival_turn
                    current_zone = next_zone
                    self.drone_schedules[drone.drone_id][current_turn] = current_zone
                    path_index += 1
                    waiting_turns = 0
                else:
                    current_turn += 1
                    waiting_turns += 1
                    if waiting_turns > 100:
                        raise DeadlockError(drone.drone_id)
                    self._reserve_zone(current_zone, current_turn, drone.drone_id)
                    self.drone_schedules[drone.drone_id][current_turn] = current_zone

    def run(self) -> None:
        """Compute paths and build the full schedule for all drones."""
        self.path_assigner()
        self._build_schedule()
