from pathlib import Path

from flyin.models import Zone
from flyin.parser import MapParser
from flyin.simulation import Scheduler


def test_capacity_conflict_forces_wait(tmp_path: Path) -> None:
    map_file = tmp_path / "map.txt"
    map_file.write_text(
        "nb_drones: 2\n"
        "start_hub: hub 0 0\n"
        "end_hub: goal 2 0\n"
        "hub: mid 1 0\n"
        "connection: hub-mid\n"
        "connection: mid-goal\n"
    )
    graph, drones = MapParser().parse(str(map_file))
    scheduler = Scheduler(graph, drones)
    scheduler.run()
    d2_schedule = scheduler.drone_schedules["D2"]
    position_at_turn_1 = d2_schedule[1]
    assert isinstance(position_at_turn_1, Zone)
    assert position_at_turn_1.name == "hub"
