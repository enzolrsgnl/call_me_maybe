from typing import cast

from ..models import Zone, Connection
from .scheduler import Scheduler


class Simulator:
    """Executes a precomputed drone schedule turn by turn and produces the simulation output."""

    def __init__(self, scheduler: Scheduler) -> None:
        """Initialize the simulator with an already computed schedule."""
        self.scheduler = scheduler
        self.graph = scheduler.graph
        self.drones = scheduler.drones

    def _connection_name(self, connection: Connection) -> str:
        """Return a display name for a connection based on its two endpoints."""
        return f"{connection.zone_a.name}-{connection.zone_b.name}"

    def _max_turn(self) -> int:
        """Return the highest turn number across all drone schedules."""
        max_turn = 0
        for drone in self.drones:
            schedule = self.scheduler.drone_schedules[drone.drone_id]
            drone_max = max(schedule.keys())
            if drone_max > max_turn:
                max_turn = drone_max
        return max_turn

    def run(self) -> list[str]:
        """Run the simulation turn by turn and return the formatted output lines."""
        output_lines: list[str] = []
        last_turn = self._max_turn()

        for drone in self.drones:
            drone.current_zone = self.graph.start_zone
            cast(Zone, self.graph.start_zone).add_drone(drone.drone_id)

        for turn in range(1, last_turn + 1):
            movements: list[str] = []
            for drone in self.drones:
                if drone.has_arrived:
                    continue
                schedule = self.scheduler.drone_schedules[drone.drone_id]
                if turn not in schedule:
                    continue

                previous_position = schedule.get(turn - 1)
                current_position = schedule[turn]

                if previous_position is current_position:
                    continue

                if drone.current_zone is not None:
                    drone.current_zone.remove_drone(drone.drone_id)

                if isinstance(current_position, Zone):
                    current_position.add_drone(drone.drone_id)
                    drone.advance(current_position)
                    movements.append(f"{drone.drone_id}-{current_position.name}")
                    if current_position.name == cast(Zone, self.graph.end_zone).name:
                        drone.arrive()
                else:
                    current_position.add_drone(drone.drone_id)
                    drone.start_transit()
                    movements.append(f"{drone.drone_id}-{self._connection_name(current_position)}")

            if movements:
                output_lines.append(" ".join(movements))

        return output_lines
