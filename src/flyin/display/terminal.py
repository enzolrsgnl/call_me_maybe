COLORS = ["\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m"]
RESET = "\033[0m"


class TerminalDisplay:
    """Display simulation output in the terminal with per-drone coloring."""

    def assign_color_to_id(self, drone_id: str) -> str:
        """Return a deterministic ANSI color code based on a drone's numeric ID."""
        number_part = drone_id[1:]
        number = int(number_part)
        color_index = number % len(COLORS)
        return COLORS[color_index]

    def show(self, lines: list[str]) -> None:
        """Display simulation output with each drone's movements colored consistently."""
        for line in lines:
            movements = line.split()
            colored_movements = []
            for movement in movements:
                drone_id = movement.split("-", 1)[0]
                color = self.assign_color_to_id(drone_id)
                colored_movements.append(f"{color}{movement}{RESET}")
            print(" ".join(colored_movements))
