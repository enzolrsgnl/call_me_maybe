from flyin.models import Graph, Drone, ZoneType, Zone, Connection
from .exceptions import ParsingError
from typing import Optional


class MapParser:
    """Parse a map file into a graph and a fleet of drones."""

    def parse(self, filepath: str) -> tuple[Graph, list[Drone]]:
        """Parse the file and validate each line against the map format."""
        try:
            with open(filepath, "r") as f:
                graph = Graph()
                drones: list[Drone] = []
                drone_number: Optional[int] = None
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()

                    if line == "" or line.startswith("#"):
                        continue

                    if line.startswith("nb_drones:"):
                        if line_number != 1:
                            raise ParsingError(
                                line_number,
                                "'nb_drones:' must be the first line of the file",
                            )
                        parts = line.split(":")
                        try:
                            drone_number = int(parts[1].strip())
                        except ValueError:
                            raise ParsingError(line_number, f"Invalid drone count: '{parts[1]}'")
                        if drone_number <= 0:
                            raise ParsingError(
                                line_number,
                                f"The drone count must be positive, got {drone_number}",
                            )

                    elif line.startswith("start_hub:"):
                        content = line[len("start_hub:"):].strip()
                        if "[" in content:
                            first_part, second_part = content.split("[", 1)
                        else:
                            first_part = content
                            second_part = ""
                        result = first_part.strip().split()
                        if len(result) != 3:
                            raise ParsingError(line_number, f"Invalid start_hub format: '{line}'")
                        name = result[0]
                        try:
                            x = int(result[1])
                            y = int(result[2])
                        except ValueError:
                            raise ParsingError(
                                line_number,
                                f"Invalid format for coordinates: '{result[1]}, {result[2]}' "
                                "should be int.",
                            )
                        metadata = self._parse_metadata(second_part, line_number)
                        zone_type_str = metadata.get("zone", "normal")
                        try:
                            zone_type = ZoneType[zone_type_str]
                        except KeyError:
                            raise ParsingError(
                                line_number,
                                f"Invalid zone type: '{zone_type_str}' should be normal, "
                                "restricted, blocked or priority",
                            )
                        color = metadata.get("color")
                        if graph.start_zone is not None:
                            raise ParsingError(line_number, "Duplicate start_hub definition")
                        zone = Zone(
                            name=name,
                            x=x,
                            y=y,
                            zone_type=zone_type,
                            max_capacity=0,
                            is_unlimited=True,
                            color=color,
                        )
                        graph.add_zone(zone)
                        graph.start_zone = zone

                    elif line.startswith("end_hub:"):
                        content = line[len("end_hub:"):].strip()
                        if "[" in content:
                            first_part, second_part = content.strip().split("[", 1)
                        else:
                            first_part = content
                            second_part = ""
                        result = first_part.strip().split()
                        if len(result) != 3:
                            raise ParsingError(line_number, f"Invalid end_hub format: '{line}'")
                        name = result[0]
                        try:
                            x = int(result[1])
                            y = int(result[2])
                        except ValueError:
                            raise ParsingError(
                                line_number,
                                f"Invalid coordinates format: '{result[1]}', "
                                f"'{result[2]}' should be int.",
                            )
                        metadata = self._parse_metadata(second_part, line_number)
                        zone_type_str = metadata.get("zone", "normal")
                        try:
                            zone_type = ZoneType[zone_type_str]
                        except KeyError:
                            raise ParsingError(
                                line_number,
                                f"Invalid zone type: '{zone_type_str}' should be normal, "
                                "restricted, blocked or priority",
                            )
                        color = metadata.get("color")
                        if graph.end_zone is not None:
                            raise ParsingError(line_number, "Duplicate end_hub definition")
                        zone = Zone(
                            name=name,
                            x=x,
                            y=y,
                            zone_type=zone_type,
                            color=color,
                            max_capacity=0,
                            is_unlimited=True,
                        )
                        graph.add_zone(zone)
                        graph.end_zone = zone

                    elif line.startswith("hub:"):
                        content = line[len("hub:"):].strip()
                        if "[" in content:
                            first_part, second_part = content.strip().split("[", 1)
                        else:
                            first_part = content
                            second_part = ""
                        result = first_part.strip().split()
                        if len(result) != 3:
                            raise ParsingError(line_number, f"Invalid hub format: '{line}'")
                        name = result[0]
                        try:
                            x = int(result[1])
                            y = int(result[2])
                        except ValueError:
                            raise ParsingError(
                                line_number,
                                f"Invalid coordinates format: '{result[1]}', "
                                f"'{result[2]}' should be int.",
                            )
                        metadata = self._parse_metadata(second_part, line_number)
                        zone_type_str = metadata.get("zone", "normal")
                        try:
                            zone_type = ZoneType[zone_type_str]
                        except KeyError:
                            raise ParsingError(
                                line_number,
                                f"Invalid zone type: '{zone_type_str}' should be normal, "
                                "restricted, priority or blocked",
                            )
                        color = metadata.get("color")
                        max_drone_str = metadata.get("max_drones", "1")
                        try:
                            max_drones = int(max_drone_str)
                        except ValueError:
                            raise ParsingError(
                                line_number,
                                f"Invalid format for max_drone: '{max_drone_str}' "
                                "should be an int",
                            )
                        if max_drones <= 0:
                            raise ParsingError(
                                line_number,
                                f"Invalid max_drone: '{max_drone_str}' should be positive",
                            )
                        zone = Zone(
                            name=name,
                            x=x,
                            y=y,
                            zone_type=zone_type,
                            color=color,
                            max_capacity=max_drones,
                            is_unlimited=False,
                        )
                        graph.add_zone(zone)

                    elif line.startswith("connection:"):
                        content = line[len("connection:"):].strip()
                        if "[" in content:
                            first_half, second_half = content.strip().split("[", 1)
                        else:
                            first_half = content
                            second_half = ""
                        zone_names = first_half.strip().split("-")
                        if len(zone_names) != 2:
                            raise ParsingError(line_number, f"Invalid connection format: '{line}'")
                        zone_a_name = zone_names[0]
                        zone_b_name = zone_names[1]
                        if zone_a_name not in graph.zones:
                            raise ParsingError(
                                line_number,
                                f"Connection references unknown zone: '{zone_a_name}'",
                            )
                        if zone_b_name not in graph.zones:
                            raise ParsingError(
                                line_number,
                                f"Connection references unknown zone: '{zone_b_name}'",
                            )
                        zone_a = graph.zones[zone_a_name]
                        zone_b = graph.zones[zone_b_name]
                        metadata = self._parse_metadata(second_half, line_number)
                        max_capacity = metadata.get("max_link_capacity", "1")
                        try:
                            max_link_capacity = int(max_capacity)
                        except ValueError:
                            raise ParsingError(
                                line_number,
                                f"Invalid max_link_capacity: '{max_capacity}' should be an int",
                            )
                        if max_link_capacity <= 0:
                            raise ParsingError(
                                line_number,
                                f"max_link_capacity must be positive, got: {max_link_capacity}",
                            )
                        connection = Connection(
                            zone_a=zone_a,
                            zone_b=zone_b,
                            max_link_capacity=max_link_capacity,
                        )
                        graph.add_connection(connection)
                    else:
                        raise ParsingError(line_number, f"unrecognized line format: '{line}'")

                if graph.start_zone is None:
                    raise ParsingError(0, "Missing required 'start_hub:' line")
                if graph.end_zone is None:
                    raise ParsingError(0, "Missing required 'end_hub:' line")
                if drone_number is None:
                    raise ParsingError(0, "Missing required 'nb_drones:' line")

                for i in range(drone_number):
                    drone = Drone(drone_id=f"D{i + 1}", current_zone=graph.start_zone)
                    drones.append(drone)

                return graph, drones

        except FileNotFoundError:
            raise FileNotFoundError(f"Map file not found: '{filepath}'")
::!!
    def _parse_metadata(self, raw: str, line_number: int) -> dict[str, str]:
        """Parse metadata tags and return them as a validated dictionary."""
        metadata: dict[str, str] = {}
        cleaned = raw.strip("]").strip()
        if cleaned == "":
            return {}
        tags = cleaned.split()
        for tag in tags:
            if "=" in tag:
                result = tag.split("=")
                if len(result) != 2:
                    raise ParsingError(line_number, f"Invalid format for the metadata: '{raw}'")
                metadata[result[0]] = result[1]
            else:
                raise ParsingError(
                    line_number,
                    f"Invalid metadata format: '{raw}' should contains '='",
                )
        return metadata
