import sys

from flyin.cli import parse_args
from flyin.parser import MapParser
from flyin.parser.exceptions import ParsingError
from flyin.simulation import Scheduler, Simulator
from flyin.simulation.exceptions import DeadlockError, ScheduleError
from flyin.pathfinding.exceptions import PathNotFoundError
from flyin.display import TerminalDisplay


def main() -> None:
    """Entry point: parse the map, run the simulation, and display the result."""
    map_path = parse_args()

    try:
        graph, drones = MapParser().parse(map_path)
    except (ParsingError, FileNotFoundError) as error:
        print(f"Error while parsing the map: {error}", file=sys.stderr)
        sys.exit(1)

    try:
        scheduler = Scheduler(graph, drones)
        scheduler.run()
    except (PathNotFoundError, ScheduleError, DeadlockError) as error:
        print(f"Error while computing the schedule: {error}", file=sys.stderr)
        sys.exit(1)

    simulator = Simulator(scheduler)
    lines = simulator.run()

    display = TerminalDisplay()
    display.show(lines)

    total_turns = len(lines)
    print(f"\nSimulation completed in {total_turns} turns.")


if __name__ == "__main__":
    main()
