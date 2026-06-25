from pathlib import Path
import json
import sys
from .models import Function, EntryPrompt
from pydantic import ValidationError


def func_parser(path_to_file: Path) -> list[Function]:
    """ This function will pars the functions files,
    and return the list of functions """
    try:
        with open(path_to_file, "r") as f:
            func_json_converted = json.load(f)
            result = []
            for elem in func_json_converted:
                to_validate = Function.model_validate(elem)
                result.append(to_validate)
    except FileNotFoundError as e:
        print(f"Error: Impossible to open the given Function's file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format for Function: {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: Invalid model for the function: {e} ")
        sys.exit(1)
    return result


def prompt_parser(path_to_file: Path) -> list[EntryPrompt]:
    """ This function will pars the entry prompt,
    and return the list of prompt """
    try:
        with open(path_to_file, "r") as f:
            prompt_json_converted = json.load(f)
            result = []
            for elem in prompt_json_converted:
                to_validate = EntryPrompt.model_validate(elem)
                result.append(to_validate)
    except FileNotFoundError as e:
        print(f"Error: Impossible to open the given EntryPrompt's file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format for EntryPrompt: {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: Invalid model for the EntryPrompt: {e}")
        sys.exit(1)
    return result
