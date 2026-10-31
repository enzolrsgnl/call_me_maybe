from pathlib import Path

from flyin.parser import MapParser
from flyin.pathfinding import Pathfinder


def test_find_path_simple(tmp_path: Path) -> None:
    map_file = tmp_path / "map.txt"
    map_file.write_text(
        "nb_drones: 1\n"
        "start_hub: hub 0 0\n"
        "end_hub: goal 2 0\n"
        "hub: mid 1 0\n"
        "connection: hub-mid\n"
        "connection: mid-goal\n"
    )
    graph, drones = MapParser().parse(str(map_file))
    assert graph.start_zone is not None
    assert graph.end_zone is not None
    path, cost = Pathfinder().find_path(graph, graph.start_zone, graph.end_zone)
    names = [z.name for z in path]
    assert names == ["hub", "mid", "goal"]
    assert cost == 2
