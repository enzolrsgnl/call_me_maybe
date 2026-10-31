from flyin.models import Graph, Zone, Connection, ZoneType


def make_zone(
    name: str,
    zone_type: ZoneType = ZoneType.normal,
    max_capacity: int = 1,
    is_unlimited: bool = False,
) -> Zone:
    return Zone(
        name=name,
        x=0,
        y=0,
        zone_type=zone_type,
        max_capacity=max_capacity,
        is_unlimited=is_unlimited,
    )


def test_add_zone_and_lookup() -> None:
    graph = Graph()
    zone = make_zone("hub")
    graph.add_zone(zone)
    assert graph.zones["hub"] is zone


def test_get_neighbors_excludes_blocked() -> None:
    graph = Graph()
    a = make_zone("a")
    b = make_zone("b", zone_type=ZoneType.blocked)
    graph.add_zone(a)
    graph.add_zone(b)
    graph.add_connection(Connection(zone_a=a, zone_b=b))
    assert graph.get_neighbors(a) == []
