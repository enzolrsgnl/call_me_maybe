import argparse
from pathlib import Path
from .reader import func_parser, prompt_parser
from llm_sdk import Small_LLM_Model
import json
import sys
from .func_chooser import func_chooser
from .generator import generate_json
from .models import OutputResult
from .writer import writer

parser = argparse.ArgumentParser()
parser.add_argument("--functions_definition",
                    default="data/input/functions_definition.json")
parser.add_argument("--input",
                    default="data/input/function_calling_tests.json")
parser.add_argument("--output",
                    default="data/output/function_calling_results.json")
args = parser.parse_args()

func_parsed = func_parser(Path(args.functions_definition))
prompt_parsed = prompt_parser(Path(args.input))

model = Small_LLM_Model()
path_vocab = model.get_path_to_vocab_file()
try:
    with open(path_vocab, "r") as f:
        vocab = json.load(f)
except FileNotFoundError as e:
    print(f"Invalid path to file: {e}")
    sys.exit(1)
except PermissionError as e:
    print(f"Permission denied: {e}")
    sys.exit(1)

result = []
for prompt in prompt_parsed:
    choosen_func = func_chooser(vocab, model,
                                prompt, func_parsed)
    generated = generate_json(model, vocab,
                              prompt, choosen_func)
    result.append(OutputResult(prompt=prompt.prompt,
                               name=choosen_func.name,
                               parameters=generated))

writer(Path(args.output), result)
