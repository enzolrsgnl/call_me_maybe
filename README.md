*This project has been created as part of the 42 curriculum by elerossi.*

# call me maybe

## Description

This project implements a **function calling system** that translates natural language prompts into structured, machine-executable function calls. Given a question such as *"What is the sum of 2 and 3?"*, the program does not answer the question directly — instead, it identifies which function should be called (`fn_add_numbers`) and with which arguments (`{"a": 2.0, "b": 3.0}`).

The core challenge of this project is that small language models (here, **Qwen3-0.6B**, a 0.6 billion parameter model) are notoriously unreliable at producing well-structured output through prompting alone. To guarantee 100% valid, schema-compliant JSON output, this project implements **constrained decoding from scratch**: at every generation step, the raw logits produced by the model are masked so that only tokens consistent with the expected JSON structure and parameter types can be selected. The model's own intelligence is preserved — it still decides what to write — but it can only express itself through a structurally valid path.

## Instructions

### Requirements

- Python 3.12+
- uv as the package manager

### Installation

```bash
make install
```

This runs `uv sync`, which creates a virtual environment and installs all dependencies declared in `pyproject.toml` (`numpy`, `pydantic`, `torch`, `transformers`, `huggingface-hub`), plus the dev dependencies (`flake8`, `mypy`).

The `llm_sdk` package (provided separately) must be present at the project root, alongside `src/`.

### Running the program

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src
```

By default, the program reads `data/input/functions_definition.json` and `data/input/function_calling_tests.json`, and writes its output to `data/output/function_calling_results.json`. All three paths can be overridden:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The first run will download the Qwen3-0.6B model weights from Hugging Face Hub (a few hundred MB), which are then cached locally.

### Debugging

```bash
make debug
```

Runs the program under Python's built-in debugger (`pdb`).

### Linting

```bash
make lint
```

Runs `flake8` and `mypy` with the flags required by the subject. `llm_sdk/` is excluded from `mypy` checks (via `mypy.ini`) since it is a third-party package outside of this project's control, and `.venv/` is excluded from `flake8` (via `.flake8`).


make lint-strict

Runs `flake8` and `mypy --strict`.

### Cleaning


make clean

Removes `__pycache__` and `.mypy_cache`.

## Resources

### Documentation and references

- Qwen3 model card
- Byte-Pair Encoding tokenization
- Pydantic v2 documentation
- Python `argparse` documentation
- Python `enum` documentation

### AI usage

AI (Claude, Anthropic) was used throughout this project as a guided learning tool, following a strict "explain, don't solve" approach: the AI was instructed to never provide ready-made code, and instead ask questions, give hints, and let the implementation be written and debugged by hand.

Concretely, AI was used for:

- **Conceptual explanations**: how LLMs work end-to-end (tokenization with Byte-Pair Encoding, embeddings, the Transformer attention mechanism, logits, softmax, autoregressive generation, greedy decoding vs. sampling) — all explained from first principles, since this was a new domain.
- **Project architecture**: deciding how to split the codebase into modules (`models.py`, `reader.py`, `generator.py`, `func_chooser.py`, `writer.py`, `__main__.py`) and designing the state machine.
- **Clarifying unfamiliar Python/library concepts**: `argparse`, Python `Enum` with `auto()`, and `numpy`'s argmax.
- **Debugging real execution bugs**, discovered only once the full pipeline was run against the actual Qwen3-0.6B model:
  - An infinite generation loop caused by comparing target strings using plain ASCII spaces, when the Qwen tokenizer actually encodes a leading space as the special character `Ġ` (e.g. `"name":Ġ"` rather than `"name": "`). This required inspecting the real vocabulary file and the model's own tokenization of a target JSON string to recover the correct fused tokens (`'Ġ"'`, `'Ġ{"'`, etc.).
  - An infinite-number-generation loop because the decoder never offered `,` or `}` as valid options after a digit, so the model could never naturally choose to stop; this was fixed by explicitly allowing the correct closing character (computed from whether more parameters remain to be generated) as a valid next token.
  - A shared-list-reference bug where `initial_tokens` and `state.current_token_sequence` pointed to the same list object, so `initial_len` ended up counting tokens generated later in the run rather than the prompt's original length.

All AI-suggested fixes were tested, verified against real model output, and understood before being accepted — debugging was driven by reading actual execution traces (the model's own generated tokens) and forming hypotheses, not by accepting suggested code blindly.

## Algorithm explanation

The implementation follows three layers:

1. **Vocabulary-level token validation** (`token_validator`, `filter_candidate`): for "closed-choice" generation steps (the JSON structure's fixed text, the function name, parameter names), the system maintains a list of valid string candidates and filters it by prefix match (`str.startswith`) as each new token extends the already-generated text. A token is valid if, once appended, the result is still a prefix of at least one remaining candidate.

2. **Grammar-based token validation** (`number_checker`, `string_checker`, `is_valid_number_token`, `is_valid_string_token`, `get_valid_tokens_for_value`): for "free" generation steps (numeric or string parameter values, where there is no fixed list of candidates), each candidate token is validated character-by-character against a small grammar (digits/`.`/`-` for numbers with at most one decimal point and an optional leading minus sign; any character for strings, framed by an opening and closing `"`). Multi-character tokens are validated by replaying the grammar check over each of their characters in sequence.

3. **Logit masking and selection** (`logits_masker`, `logit_chooser`): once the valid token IDs are known, every other position in the model's raw logit vector is set to negative infinity, and the highest-scoring remaining token is selected (`argmax`, implemented as `logits.index(max(logits))`). Because invalid tokens have probability zero after this masking, the generated sequence is guaranteed to always be valid JSON, regardless of what the underlying model "wants" to generate.

A finite-state machine (`Step` enum + `GenerationState` class) tracks progress through nine states (`OPEN_BRACE`, `NAME_KEY`, `FUNCTION_NAME`, `PARAMETERS_KEY`, `PARAM_NAME`, `PARAM_VALUE`, `SEPARATOR`, `CLOSING_PARAM`, `CLOSING_BRACE`), with explicit transition logic (`get_next_step`, `transitionner`) handling both straightforward fixed-text transitions and the conditional looping needed for functions with multiple parameters.

Function selection (`func_chooser`) and argument extraction (`generate_json`) both reuse this same constrained-decoding machinery, but operate on enriched prompts that include the available function(s)' name, description, and parameter list, so that the underlying 0.6B model has enough context to make a meaningful choice rather than guessing blindly.

## Design decisions

- **No external constrained-decoding library** (`outlines` and similar are forbidden by the subject): the entire token validation and masking pipeline was built from the raw vocabulary file and `get_logits_from_input_ids`, as required.
- **Pure Python `argmax`** instead of `numpy.argmax`, to avoid converting between Python lists and numpy arrays for a single operation that a one-line list method already covers.
- **Separation of "structural" and "free" generation**: rather than a single monolithic validator, the fixed JSON skeleton (closed-choice, prefix-filtered) and the free-form parameter values (grammar-checked) are handled by distinct functions, keeping each piece testable in isolation.
- **Enriched prompts for both function selection and argument extraction**: since the model otherwise has no way to know which functions exist or what a given parameter represents, both `func_chooser` and `generate_json` build a short natural-language context block (function name, description, parameters) before the actual user question.
- **Fail-fast error handling**: input file errors (`FileNotFoundError`, `json.JSONDecodeError`, Pydantic's `ValidationError`) print a clear message and exit with status 1, since the program cannot meaningfully continue without valid function definitions or prompts.

## Performance analysis

- **Validity**: 100% of generated outputs are syntactically valid, schema-compliant JSON, by construction — invalid tokens can never be selected, regardless of model behavior.
- **Accuracy**: across the 11 provided test prompts, function selection and argument extraction were correct for the simple arithmetic, greeting, string-reversal, and square-root cases. Extraction quality degrades on prompts involving regex substitution with three consecutive string parameters, where the small model sometimes fails to extract meaningful values for all three fields — a known limitation of using a 0.6B parameter model for multi-slot extraction, rather than a constrained-decoding bug.
- **Speed**: running entirely on CPU (no compatible GPU available in the development environment), a single prompt with 2-3 short parameters takes from a few seconds to about a minute, depending on parameter type and length; the full 11-prompt test set takes several minutes. On GPU hardware, generation would be substantially faster.

### Regex parameters and model limits

Parameters named `regex`, `regexp`, or `pattern` use a dedicated constrained
grammar. It permits short literal patterns and character classes (including
ranges and negation), while rejecting wildcards, alternation, and quantifiers. The prompt
also tells the model to translate semantic categories such as vowels, digits,
or spaces into a character class. This is parameter-role detection, independent
of the input test file; function selection remains entirely model-driven.

This improves reliability but cannot make semantic accuracy absolute. In a
request such as "all vowels", the expected pattern does not occur literally in
the input: the model must generalize from language to regex. Constrained
decoding guarantees that the result belongs to the supported regex grammar and
that the enclosing JSON is valid; it cannot prove that a syntactically valid
class has the intended meaning. A 0.6B model can therefore still emit a valid
but semantically wrong class. Perfect reliability would require an explicit
concept-to-pattern knowledge table, a symbolic component, or a stronger model.

## Challenges faced

The single biggest challenge was discovering that **Qwen's tokenizer does not use a plain ASCII space character** — it encodes a leading space as the special Unicode character `Ġ` (`\u0120`), fused into the following token (e.g. `'Ġ"'`, `'Ġ{"'`). Every fixed-text target string in the JSON skeleton (`"name": "`, `, "parameters": {`, etc.) initially used regular spaces, which meant no token in the real vocabulary could ever satisfy the required prefix — causing the decoder to fall back to selecting an arbitrary token (since `max()` over an all `-inf` logit array still returns a value, with `.index()` defaulting to position 0). This was diagnosed by encoding a hand-written example of the exact target JSON with the real tokenizer and inspecting which tokens it actually produced.

A second significant challenge was making the model voluntarily *stop* generating digits for a numeric parameter: since digits were always valid tokens, nothing in the original implementation ever pushed the model towards closing the number. The fix required explicitly adding the contextually correct closing character (`,` or `}`, computed from whether the current parameter is the last one) to the set of valid next tokens, alongside the usual digit/`.`/`-` grammar.

A related and more subtle issue arose when a single multi-character token (e.g. `'"}}'`) could pass character-by-character grammar validation for a string value while embedding structural JSON characters *after* its closing quote — solved by rejecting any candidate token where the closing `"` does not fall on the token's very last character.

Finally, the project required building two separate "enriched prompt" mechanisms (for function selection and for argument extraction) once it became clear that giving the model only the bare user question, with no information about available functions or expected parameters, led to systematically wrong function choices and meaningless parameter values.

## Testing strategy

Testing was done incrementally at every layer:

- Each pure function (`filter_candidate`, `token_validator`, `logits_masker`, `logit_chooser`, `number_checker`, `string_checker`, `get_next_step`, `get_candidat`, etc.) was first tested in isolation via one-off `uv run python3 -c "..."` snippets, with hand-constructed inputs, before being wired into the full pipeline.
- The full pipeline was then tested end-to-end against the real Qwen3-0.6B model, starting with a single isolated prompt (via `--input` pointing to a small custom JSON file) to keep iteration cycles short, before progressively testing batches of prompts and finally the full 11-prompt test set.
- Debug `print` statements (temporarily added inside the generation loop and the transition logic) were used to trace, token by token, exactly which state the generator was in and what text had been accumulated, which was essential to diagnose the tokenizer-encoding and infinite-loop bugs described above. These were removed before final submission.
- `flake8` and `mypy --strict` were run after every change to every file, to catch type and style issues immediately rather than at the end.

The regex constraint unit tests can be run with:

```bash
uv run python -m unittest discover -s tests
```

The three end-to-end regex scenarios (literal word, semantic vowels, and the
similar-but-distinct digits case) can be run with:

```bash
uv run python -m src --input data/input/test_regex_cases.json \
  --output data/output/test_regex_cases.json
```

## Example usage

```bash
$ make install
$ make run
```

Given this `data/input/function_calling_tests.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"}
]
```

and the corresponding `functions_definition.json`, the resulting `data/output/function_calling_results.json` looks like:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```
