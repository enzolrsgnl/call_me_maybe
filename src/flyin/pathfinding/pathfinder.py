import heapq
from flyin.models import Graph, Zone, ZoneType
from .exceptions import PathNotFoundError


class Pathfinder:
    """Compute shortest weighted paths between zones using Dijkstra's algorithm."""

    def find_path(self, graph: Graph, start: Zone, end: Zone) -> tuple[list[Zone], int]:
        """Return the optimal path and its total cost between two zones."""
        distances: dict[str, int] = {start.name: 0}
        previous: dict[str, Zone] = {}
        queue: list[tuple[int, int, str]] = [(0, 0, start.name)]
        visited: set[str] = set()
        priority_count: dict[str, int] = {start.name: 0}

        while queue:
            current_cost, neg_priority, current_name = heapq.heappop(queue)
            if current_name in visited:
                continue
            visited.add(current_name)
            if current_name == end.name:
                break
            current_zone = graph.zones[current_name]
            all_neighbors = graph.get_neighbors(current_zone)
            for neighbor_zone, connection, connection_cost in all_neighbors:
                if neighbor_zone.zone_type == ZoneType.priority:
                    is_priority = True
                else:
                    is_priority = False
                current_priority_count = priority_count[current_name]
                new_priority_count = current_priority_count + (1 if is_priority is True else 0)
                new_cost = current_cost + connection_cost
                known_cost = distances.get(neighbor_zone.name)
                known_priority = priority_count.get(neighbor_zone.name)
                is_better_cost = known_cost is None or known_cost > new_cost
                is_tied_with_more_priority = (
                    new_cost == known_cost
                    and new_priority_count > (known_priority or 0)
                )
                if is_better_cost or is_tied_with_more_priority:
                    distances[neighbor_zone.name] = new_cost
                    previous[neighbor_zone.name] = current_zone
                    heapq.heappush(queue, (new_cost, -new_priority_count, neighbor_zone.name))
                    priority_count[neighbor_zone.name] = new_priority_count
        if end.name not in distances:
            raise PathNotFoundError(start.name, end.name)

        path: list[Zone] = []
        current_zone_tracked = end
        while current_zone_tracked.name != start.name:
            path.append(current_zone_tracked)
            current_zone_tracked = previous[current_zone_tracked.name]
        path.append(start)
        path.reverse()

        return path, distances[end.name]
