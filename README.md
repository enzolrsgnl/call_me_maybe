*This project has been created as part of the 42 curriculum by elerossi.*

# call me maybe

## Description

A function-calling tool that takes a natural language question and returns the right function to call with the right arguments as structured JSON, not a text answer.

Given `"What is the sum of 2 and 3?"`, the output is:
```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The core challenge: a 0.6B parameter model is unreliable at producing valid JSON from prompting alone. To guarantee 100% schema-compliant output, the project implements constrained decoding, at every generation step, any token that would break the JSON structure or violate the schema is masked to -inf before token selection. The model decides what to say, it just can't say anything structurally invalid.

## Instructions

# Install dependencies
make install

# Run (reads data/input/, writes data/output/)
make run

# Or with explicit paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Debug (pdb)
make debug

# Lint (flake8 + mypy)
make lint

# Remove caches
make clean
```

The first run downloads Qwen3-0.6B model weights from Hugging Face Hub (~few hundred MB, cached after that).

The `llm_sdk/` directory must be present at the project root alongside `src/`.

## Algorithm explanation

The generation pipeline works in two steps, both using constrained decoding:

1. **Function selection** (func_chooser.py): the model picks a function name from the list of candidates. An enriched prompt includes all available function names and descriptions. At each generation step, only tokens that are a valid prefix of one of the function names are allowed.

2. **Argument extraction** (generator.py): a finite state machine drives generation through nine states (`OPEN_BRACE → NAME_KEY → FUNCTION_NAME → PARAMETERS_KEY → PARAM_NAME → PARAM_VALUE → ... → CLOSING_BRACE`). For each state, only structurally valid tokens can be selected:
   - Fixed JSON skeleton (braces, keys, separators): prefix-filtered against the exact target string.
   - Numeric values: character-by-character grammar — digits, one optional `.`, optional leading `-`. Candidate values are extracted from the prompt with a regex to avoid infinite digit generation.
   - String values: constrained to JSON-quoted strings extracted from the prompt. Falls back to free generation with a 30-character cap.
   - Boolean values: constrained to `true` / `false`.

At every step: get logits → set invalid token logits to `-inf` → pick highest remaining logit (greedy).

## Design decisions

- **No external constrained-decoding library** — the subject forbids outlines and similar. Everything is built from the raw vocab file and get_logits_from_input_ids.
- **State machine over a single validator** — separating fixed-text states from free-value states keeps each piece small and testable independently.
- **Numbers extracted from the prompt** — rather than letting the model generate digits freely (which causes infinite loops), numbers are extracted from the original question and used as a closed candidate list.
- **Regex parameters detected by name/description** — parameters named regex, regexp, or pattern trigger a dedicated grammar that allows short literals and character classes, preventing the model from emitting wildcards or quantifiers.

## Performance analysis

- **JSON validity**: 100% invalid tokens are structurally impossible to select.
- **Accuracy**: correct on all simple cases (arithmetic, greetings, string reversal, square root). Multi-slot regex extraction degrades slightly with the 0.6B model, which is a model-size limitation, not a constrained-decoding bug.
- **Speed**: a few seconds per prompt on GPU; up to a minute per prompt on CPU.

## Challenges faced

The hardest bug: Qwen's tokenizer represents a leading space as `Ġ` (`Ġ`), not a plain ASCII space. Every fixed-text target (`"name": "`, `, "parameters": {`) was initially written with regular spaces, so no vocabulary token ever matched — the decoder silently fell back to token 0 on every step. Fixed by encoding a target JSON string with the actual tokenizer and inspecting the output to recover the correct fused tokens (`'Ġ"'`, `'Ġ{"'`, etc.).

A second issue: nothing ever triggered the model to stop generating digits. Fixed by adding the contextually correct closing character (`,` or `}`) as a valid next token alongside the digit grammar.

## Testing strategy

Each pure function was tested in isolation with `uv run python3 -c "..."` snippets before being integrated. The full pipeline was then validated end-to-end against the real model, one prompt at a time, using temporary `--input` files to keep iteration cycles short. `flake8` and `mypy --strict` were run after every change.

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Constrained decoding — overview](https://huggingface.co/blog/constrained-beam-search)

### AI usage

Claude was used for conceptual explanations (how tokenization, logits, and autoregressive generation work), help designing the state machine architecture, and debugging particularly the Ġ space tokenizer bug, the regex format, the infinite digit-generation loop and this readme. All suggested fixes were tested against real model output and rewritten by hand before being committed. 

