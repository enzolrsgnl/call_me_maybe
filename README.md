*This project has been created as part of the 42 curriculum by elerossi.*

## Description

Fly-in simulates drone routing through a network of connected zones (hubs), from a start hub to an end hub, aiming to minimize the number of tours needed to move every drone. Zones and connections have limited capacity, and some zones carry special rules: restricted (one drone at a time, atomically), priority (served first), or blocked (impassable). The project is implemented 100% in Object-Oriented Python, without relying on any external graph library — all graph structures and algorithms are built from scratch.

## Instructions

Install dependencies:

```
make install
```

Run the simulation on a map:

```
make run MAP=<path>
```

`MAP` must point to a map file describing the number of drones, the start/end hubs, intermediate hubs, and the connections between them. Example using one of the provided maps:

```
make run MAP=maps/test_normal.txt
```

Run with the Python debugger attached:

```
make debug MAP=maps/test_normal.txt
```

Other targets:

```
make clean         # remove __pycache__, .mypy_cache, .pytest_cache
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
```

## Resources

- Claude used for debugging, explaining concepts (Dijkstra, heapq), and reviewing code throughout development.
- General YouTube videos on Dijkstra's algorithm and priority queues (heapq) to understand the concepts before implementing them.

## Algorithm & Implementation Choices

- **Pathfinding**: weighted Dijkstra using `heapq`, with tie-breaking for priority zones via a negative counter pushed onto the heap.
- **Scheduling**: prioritized planning with a reservation table keyed by (tour, zone) and (tour, connection); the restricted case (two tours) is handled atomically; a deadlock guard caps consecutive waiting tours per drone.
- **Architecture**: layered design — `models` (Zone, Drone, Connection, Graph), `parser`, `pathfinding`, `simulation` (scheduler + simulator), and `display`.

## Visual Representation

The simulation is displayed in a colored terminal, with each drone assigned a deterministic color derived from its ID. This makes it possible to visually follow a given drone across tours, even when several drones move on the same line at once.
