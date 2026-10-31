import argparse
from typing import cast


def parse_args() -> str:
    """Parse command-line arguments and return the path to the map file."""
    parser = argparse.ArgumentParser(
        prog="flyin",
        description="Simulate drone routing through a network of connected zones.",
    )
    parser.add_argument(
        "map_path",
        type=str,
        help="Path to the map file describing zones, connections, and drones.",
    )
    args = parser.parse_args()
    return cast(str, args.map_path)
