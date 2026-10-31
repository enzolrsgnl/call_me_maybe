from pathlib import Path

from flyin.parser import MapParser


def test_parse_basic_map(tmp_path: Path) -> None:
    map_file = tmp_path / "map.txt"
    map_file.write_text(
        "nb_drones: 1\n"
        "start_hub: hub 0 0\n"
        "end_hub: goal 1 0\n"
        "connection: hub-goal\n"
    )
    graph, drones = MapParser().parse(str(map_file))
    assert graph.start_zone is not None
    assert graph.end_zone is not None
    assert graph.start_zone.name == "hub"
    assert graph.end_zone.name == "goal"
    assert len(drones) == 1
