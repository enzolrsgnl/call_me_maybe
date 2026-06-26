from pathlib import Path
import json
from .models import OutputResult
import sys


def writer(path: Path, output: list[OutputResult]) -> None:
    """ This function write the content in the output file in json format """
    result = []
    for elem in output:
        result.append(elem.model_dump())
    try:
        with open(path, "w") as f:
            json.dump(result, f)
    except FileNotFoundError as e:
        print(f"Invalid path file: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Permission denied: {e}")
        sys.exit(1)
