import math
from enum import Enum, auto
from .models import Function, EntryPrompt, OutputResult
from llm_sdk import Small_LLM_Model
import json
import re

# BPE tokenizers encode a preceding space as this character (U+0120).
# Generated token strings use it, but candidates extracted from plain text use
# regular spaces — so we must normalize before comparing the two.
_VOCAB_SPACE = 'Ġ'
_MAX_REGEX_LENGTH = 24


def _to_plain(s: str) -> str:
    """ Convert vocabulary token representation to plain text (Ġ → space).
    Used only when comparing generated text against plain-text candidates. """
    return s.replace(_VOCAB_SPACE, ' ')


def filter_candidate(candidates: list[str],
                     already_generated: str) -> list[str]:
    """ This function is a filter dedicated to keep the corresponding
    elements of the list with the text already generated """
    candidates_left = []
    for elem in candidates:
        if elem.startswith(already_generated):
            candidates_left.append(elem)
    return candidates_left


def token_validator(vocab: dict[str, int], already_generated: str,
                    remaining_candidates: list[str]) -> list[int]:
    """ This function add the id of all valids token to a list,
        the other token that are not in this list will be marked -inf """
    token_valids = []
    for token_string, token_id in vocab.items():
        token_to_test = already_generated + token_string
        candidates_pool = filter_candidate(remaining_candidates, token_to_test)
        if candidates_pool:
            token_valids.append(token_id)
    return token_valids


def logits_masker(valids_token: list[int], logits: list[float]) -> list[float]:
    """ This function is the wrong logit masker, it mark them as -inf,
    then the good logits keep their value
    and the wrong ones are dodged for the generation"""
    masked_logit = []
    valid_ids = set(valids_token)
    for index, logit in enumerate(logits):
        if index not in valid_ids:
            logit = -math.inf
        masked_logit.append(logit)
    return masked_logit


def logit_chooser(logits: list[float]) -> int:
    """ this function choose the best logit
    with the higher score to be generated """
    choosen = logits.index(max(logits))
    return choosen


class Step(Enum):
    OPEN_BRACE = auto()
    NAME_KEY = auto()
    FUNCTION_NAME = auto()
    PARAMETERS_KEY = auto()
    PARAM_NAME = auto()
    PARAM_VALUE = auto()
    SEPARATOR = auto()
    CLOSING_PARAM = auto()
    CLOSING_BRACE = auto()


class GenerationState():
    """ This class is a state class that change in function
    of the generated content """
    def __init__(self, current_state: Step, current_function: Function,
                 current_param_name: str, current_generated_text: str,
                 current_token_sequence: list[int],
                 treated_params: list[str],
                 remaining_string_candidates: list[str],
                 literal_regex_candidates: list[str]) -> None:
        self.current_state = current_state
        self.current_function = current_function
        self.current_param_name = current_param_name
        self.current_generated_text = current_generated_text
        self.current_token_sequence = current_token_sequence
        self.treated_params = treated_params
        # Quoted-phrase candidates extracted from the prompt ('...').
        # Consumed one by one as string parameters are filled in order to
        # prevent the same candidate from being assigned to two parameters.
        self.remaining_string_candidates = remaining_string_candidates
        self.literal_regex_candidates = literal_regex_candidates


step_target = {Step.OPEN_BRACE: '{', Step.NAME_KEY: '\"name\":Ġ\"',
               Step.PARAMETERS_KEY: '",Ġ\"parameters\":Ġ{\"',
               Step.SEPARATOR: ',Ġ\"',
               Step.CLOSING_PARAM: '}', Step.CLOSING_BRACE: '}'}


def get_function_name(functions: list[Function]) -> list[str]:
    """ This function allow to get the name of the current fonction treated """
    result = []
    for f in functions:
        result.append(f.name)
    return result


def number_checker(candidate_char: str, already_generated: str) -> bool:
    """ This function is a position validator for a type number
    parameter with 3 simple rules """
    if not already_generated and candidate_char == "-":
        return True
    elif (already_generated and candidate_char == "."
          and "." not in already_generated):
        return True
    elif candidate_char.isdigit():
        return True
    else:
        return False


def string_checker(candidate_char: str, already_generated: str) -> bool:
    """ This function is a string parameter validator with a simple rule """
    if already_generated == "" and candidate_char != '"':
        return False
    elif candidate_char == '"' and len(already_generated) == 1:
        return False
    elif candidate_char in ",{}" and len(already_generated) == 1:
        return False
    elif candidate_char == "\\":
        return False
    elif ord(candidate_char) < 32 or 0x0100 <= ord(candidate_char) <= 0x011F:
        return False
    else:
        return True


def get_next_step(current_state: Step,
                  current_function: Function,
                  treated_params: list[str]) -> Step:
    """ This function is the decisioner of to moove to the next step or not """
    if current_state == Step.OPEN_BRACE:
        return Step.NAME_KEY
    elif current_state == Step.NAME_KEY:
        return Step.FUNCTION_NAME
    elif current_state == Step.FUNCTION_NAME:
        return Step.PARAMETERS_KEY
    elif current_state == Step.PARAMETERS_KEY:
        return Step.PARAM_NAME
    elif current_state == Step.PARAM_NAME:
        return Step.PARAM_VALUE
    elif current_state == Step.PARAM_VALUE:
        if len(treated_params) == len(current_function.parameters):
            return Step.CLOSING_PARAM
        else:
            return Step.SEPARATOR
    elif current_state == Step.SEPARATOR:
        return Step.PARAM_NAME
    elif current_state == Step.CLOSING_PARAM:
        return Step.CLOSING_BRACE
    else:
        raise ValueError(f"Unexpected state: {current_state}")


def extract_numbers_from_prompt(text: str) -> list[str]:
    """ This function extracts all numbers present in the original
    prompt text, to use as a closed candidate list for number values """
    return re.findall(r'-?\d+\.?\d*', text)


def extract_quoted_phrases(text: str) -> list[str]:
    """Extract strings enclosed by matching single or double quotes.

    A double-quoted source may contain an apostrophe (for example ``I'm``),
    so the opening delimiter is captured and only its matching delimiter can
    close the phrase. Order is preserved and duplicates are removed.
    """
    matches = re.findall(r'(["\'])(.*?)\1', text)
    phrases = [phrase for _, phrase in matches]
    return list(dict.fromkeys('"' + phrase + '"' for phrase in phrases))


def extract_literal_regex_candidates(text: str) -> list[str]:
    """Extract explicitly named literal targets for a regex parameter.

    This recognizes a linguistic role (word/text/string to replace), not a
    particular value. General categories such as vowels or digits deliberately
    produce no candidate and remain generated by the regex grammar.
    """
    patterns = (
        r"\b(?:word|text|string)\s+([\"'])(.*?)\1",
        r"\b(?:replace|substitute)\s+([\"'])(.*?)\1",
    )
    candidates: list[str] = []
    for pattern in patterns:
        for _, value in re.findall(pattern, text, flags=re.IGNORECASE):
            candidate = '"' + value + '"'
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def get_candidat(current_state: Step,
                 current_function: Function,
                 treated_params: list[str]) -> list[str]:
    """ This function get the list of valid candidates
    according to the state """
    if current_state == Step.OPEN_BRACE:
        return [step_target[current_state]]
    elif current_state == Step.NAME_KEY:
        return [step_target[current_state]]
    elif current_state == Step.PARAMETERS_KEY:
        return [step_target[current_state]]
    elif current_state == Step.SEPARATOR:
        return [step_target[current_state]]
    elif current_state == Step.CLOSING_PARAM:
        return [step_target[current_state]]
    elif current_state == Step.FUNCTION_NAME:
        return get_function_name([current_function])
    elif current_state == Step.PARAM_NAME:
        remaining = []
        for elem in current_function.parameters.keys():
            if elem not in treated_params:
                remaining.append(elem + '":')
        return remaining
    elif current_state == Step.CLOSING_BRACE:
        return [step_target[current_state]]
    else:
        raise ValueError(f"Unexpected state: {current_state}")


def is_valid_number_token(token_string: str, already_generated: str) -> bool:
    """ This function use number_checker to test if the token sequence
    is valid when its a number """
    generated_updated = already_generated
    for c in token_string:
        if number_checker(c, generated_updated):
            generated_updated = generated_updated + c
        else:
            return False
    return True


def is_valid_string_token(token_string: str, already_generated: str) -> bool:
    """ This function use string_checker to test if the token sequence
    is valid when its a string """
    generated_updated = already_generated
    for index, c in enumerate(token_string):
        if string_checker(c, generated_updated):
            generated_updated = generated_updated + c
            if (
                c == '"' and len(generated_updated) > 1
                and index != len(token_string) - 1
            ):
                return False
        else:
            return False
    return True


def is_regex_parameter(param_name: str, function: Function) -> bool:
    """Return whether a string parameter is intended to hold a regex.

    Function schemas only expose names and a function-level description, so
    both are used.  This recognizes a semantic role, not a particular prompt
    or test value.
    """
    normalized_name = param_name.lower().replace("-", "_")
    name_parts = normalized_name.split("_")
    regex_words = {"regex", "regexp", "pattern"}
    if regex_words.intersection(name_parts):
        return True
    description = function.description.lower()
    return "regular expression" in description and "pattern" in name_parts


def _is_complete_simple_regex(body: str) -> bool:
    """Accept the deliberately small regex language emitted by the decoder."""
    if re.fullmatch(r"[A-Za-z0-9 _-]+", body) is not None:
        return True
    return re.fullmatch(r"\[\^?[A-Za-z0-9 _-]+\]", body) is not None


def is_valid_regex_token(token_string: str,
                         already_generated: str) -> bool:
    """Validate a token as a prefix of a simple, JSON-quoted regex.

    Supported regexes are literals and character classes (including ranges
    and negated classes). Excluding wildcards, alternation, and quantifiers
    keeps a tiny model from choosing an overly broad or meaningless pattern.
    """
    candidate = _to_plain(already_generated + token_string)
    if not candidate.startswith('"') or len(candidate) == 1:
        return candidate == '"'
    if candidate.count('"') > 2:
        return False
    if '"' in candidate[1:-1]:
        return False

    closed = candidate.endswith('"')
    body = candidate[1:-1] if closed else candidate[1:]
    if len(body) > _MAX_REGEX_LENGTH:
        return False
    if closed:
        return _is_complete_simple_regex(body)
    if body == "":
        return True
    if body.startswith("["):
        if body.count("[") != 1 or body.count("]") > 1:
            return False
        if "]" in body:
            return body.endswith("]") and _is_complete_simple_regex(body)
        class_body = body[1:]
        if class_body.startswith("^"):
            class_body = class_body[1:]
        return all(char.isascii() and (char.isalnum() or char in " _-")
                   for char in class_body)
    return all(char.isascii() and (char.isalnum() or char in " _-")
               for char in body)


def _get_regex_tokens(vocab: dict[str, int],
                      already_generated: str) -> list[int]:
    """Return vocabulary ids that keep generation in the regex grammar."""
    return [
        token_id
        for token_string, token_id in vocab.items()
        if is_valid_regex_token(token_string, already_generated)
    ]


def _get_plain_candidate_tokens(vocab: dict[str, int],
                                already_generated: str,
                                candidates: list[str]) -> list[int]:
    """Match plain-text candidates against tokenizer space markers."""
    plain_generated = _to_plain(already_generated)
    return [
        token_id
        for token_string, token_id in vocab.items()
        if any(candidate.startswith(
            plain_generated + _to_plain(token_string)
        ) for candidate in candidates)
    ]


def _get_free_string_tokens(vocab: dict[str, int],
                            already_generated: str) -> list[int]:
    """ Free JSON-string token ids: any JSON-safe chars up to 30 total,
    then only the closing double-quote. Used as a fallback when no
    quoted-phrase candidate from the prompt matches the current parameter.
    Cap at 30 chars (not 50) to limit LLM inference calls for free params
    like regex patterns that the model tends to make very long. """
    valid: list[int] = []
    for token_string, token_id in vocab.items():
        if len(already_generated) >= 30:
            if token_string == '"':
                valid.append(token_id)
        else:
            if is_valid_string_token(token_string, already_generated):
                valid.append(token_id)
    return valid


def get_valid_tokens_for_value(vocab: dict[str, int], already_generated: str,
                               param_type: str, treated_params: list[str],
                               parameters_number: int,
                               string_candidates: list[str]) -> list[int]:
    """ Token validator for the PARAM_VALUE state (string parameters).

    Strategy: constrain the LLM to pick from the remaining quoted-phrase
    candidates extracted from the prompt. Once those are exhausted (e.g.
    for a regex or replacement that is not literally quoted), fall back to
    free JSON-string generation with a 50-character safety cap.

    Normalization: vocabulary tokens use Ġ for spaces, but candidates are
    plain text — _to_plain converts Ġ→space before every comparison. """
    if param_type != "string":
        return []
    plain_gen = _to_plain(already_generated)
    remaining = [c for c in string_candidates if c.startswith(plain_gen)]
    if remaining:
        # Constrained: only tokens that advance toward at least one candidate.
        valid: list[int] = []
        for token_string, token_id in vocab.items():
            test = plain_gen + _to_plain(token_string)
            if any(c.startswith(test) for c in remaining):
                valid.append(token_id)
        return valid
    # Fallback: free generation (no matching candidate left).
    return _get_free_string_tokens(vocab, already_generated)


def transitionner(state: GenerationState, prompt_text: str) -> None:
    """ This function allow the state transition comparing
    the generated text and the target """
    if state.current_state in [Step.OPEN_BRACE, Step.NAME_KEY,
                               Step.PARAMETERS_KEY, Step.SEPARATOR,
                               Step.CLOSING_PARAM]:
        if state.current_generated_text == step_target[state.current_state]:
            state.current_state = get_next_step(state.current_state,
                                                state.current_function,
                                                state.treated_params)
            state.current_generated_text = ""
    elif state.current_state == Step.FUNCTION_NAME:
        if state.current_generated_text == state.current_function.name:
            state.current_state = get_next_step(state.current_state,
                                                state.current_function,
                                                state.treated_params)
            state.current_generated_text = ""
    elif state.current_state == Step.PARAM_NAME:
        candidates = get_candidat(state.current_state, state.current_function,
                                  state.treated_params)
        remaining = filter_candidate(candidates,
                                     state.current_generated_text)
        if (len(remaining) == 1
           and remaining[0] == state.current_generated_text):
            state.current_state = get_next_step(state.current_state,
                                                state.current_function,
                                                state.treated_params)
            state.current_generated_text = ""
            cleaned = remaining[0][:-2]
            state.current_param_name = cleaned
    elif state.current_state == Step.PARAM_VALUE:
        param_type = (
            state.current_function.parameters[state.current_param_name].type
        )
        if param_type == "string":
            if is_regex_parameter(state.current_param_name,
                                  state.current_function):
                plain_text = _to_plain(state.current_generated_text)
                literal_complete = (
                    plain_text in state.literal_regex_candidates
                )
                grammar_complete = (
                    not state.literal_regex_candidates
                    and len(plain_text) >= 3
                    and plain_text.endswith('"')
                    and _is_complete_simple_regex(plain_text[1:-1])
                )
                if literal_complete or grammar_complete:
                    state.treated_params.append(state.current_param_name)
                    state.current_state = get_next_step(
                        state.current_state,
                        state.current_function,
                        state.treated_params
                    )
                    state.current_generated_text = ""
                return
            # Normalize Ġ→space so vocab-form generated text can be compared
            # against the plain-text candidates extracted from the prompt.
            plain_gen = _to_plain(state.current_generated_text)
            str_remaining = [
                c for c in state.remaining_string_candidates
                if c.startswith(plain_gen)
            ]
            if (str_remaining and len(str_remaining) == 1
                    and str_remaining[0] == plain_gen):
                # Exact match: the LLM chose this candidate. Consume it so
                # subsequent parameters cannot reuse the same value.
                state.remaining_string_candidates.remove(str_remaining[0])
                state.treated_params.append(state.current_param_name)
                state.current_state = get_next_step(state.current_state,
                                                    state.current_function,
                                                    state.treated_params)
                state.current_generated_text = ""
            elif not str_remaining:
                # No candidate matches: we are in free-generation fallback.
                # Transition as soon as the JSON string is syntactically closed
                # (opening quote + at least one char + closing quote).
                text = state.current_generated_text
                if len(text) >= 3 and text[0] == '"' and text[-1] == '"':
                    state.treated_params.append(state.current_param_name)
                    state.current_state = get_next_step(state.current_state,
                                                        state.current_function,
                                                        state.treated_params)
                    state.current_generated_text = ""
        else:
            # For numbers and booleans, use the closed candidate list approach.
            cands: list[str]
            if param_type == "number":
                cands = extract_numbers_from_prompt(prompt_text)
            else:
                cands = ["true", "false"]
            rem = filter_candidate(cands, state.current_generated_text)
            if (rem and len(rem) == 1
                    and rem[0] == state.current_generated_text):
                state.treated_params.append(state.current_param_name)
                state.current_state = get_next_step(state.current_state,
                                                    state.current_function,
                                                    state.treated_params)
                state.current_generated_text = ""


def is_generation_complete(state: GenerationState) -> bool:
    """ This function is just an helper for the transitionner to check if
    the actual reinitialised state is already the end character """
    if (
        state.current_state == Step.CLOSING_BRACE
        and state.current_generated_text == step_target[Step.CLOSING_BRACE]
    ):
        return True
    else:
        return False


def generate_json(model: Small_LLM_Model,
                  vocab: dict[str, int],
                  prompt: EntryPrompt,
                  function: Function) -> dict[str, float | str | bool]:
    """ This is the most important function of the file,
    its the principal loop to generate the json format token by token """
    params_description = ""
    for param_name, param_info in function.parameters.items():
        role = ""
        if (param_info.type == "string"
                and is_regex_parameter(param_name, function)):
            role = (
                "; regex pattern: translate the requested match into one "
                "short literal or character class"
            )
        params_description = (
            params_description
            + f"- {param_name} ({param_info.type}{role})\n"
        )

    enriched_prompt = (
        f"Extract the function call arguments from the question.\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters:\n{params_description}"
        f"Regex rules: infer the pattern from the question. Prefer a plain "
        f"literal for an explicitly quoted word or text, or one character "
        f"class for a general category. If the question says to replace or "
        f"substitute the word/text 'X', the regex must be the literal X; do "
        f"not turn its letters into a character class. For "
        f"example, digits become [0-9], spaces become [ ], and vowels become "
        f"[aeiouAEIOU]. Do not use alternation or quantifiers.\n"
        f"Question: {prompt.prompt}\n"
        f"JSON:"
    )

    encoded = model.encode(enriched_prompt)
    initial_tokens = encoded[0].tolist()
    initial_len = len(initial_tokens)

    # Extract quoted phrases from the prompt (text between '...') as closed
    # string candidates. These will be consumed one by one as string params
    # are filled, ensuring each gets a distinct value from the prompt text.
    string_candidates = extract_quoted_phrases(prompt.prompt)
    literal_regex_candidates = extract_literal_regex_candidates(prompt.prompt)

    state = GenerationState(Step.OPEN_BRACE, function, "", "",
                            list(initial_tokens), [], string_candidates,
                            literal_regex_candidates)
    id_to_str = {v: k for k, v in vocab.items()}

    number_candidates = extract_numbers_from_prompt(prompt.prompt)

    while (state.current_state != Step.CLOSING_BRACE
           or state.current_generated_text != step_target[Step.CLOSING_BRACE]):
        logits = model.get_logits_from_input_ids(state.current_token_sequence)

        if state.current_state == Step.PARAM_VALUE:
            current_param = (
                state.current_function.parameters[state.current_param_name]
            )
            param_type = current_param.type
            if param_type == "number":
                valid_ids = token_validator(vocab,
                                            state.current_generated_text,
                                            number_candidates)
            elif param_type == "string":
                if is_regex_parameter(state.current_param_name, function):
                    if state.literal_regex_candidates:
                        valid_ids = _get_plain_candidate_tokens(
                            vocab,
                            state.current_generated_text,
                            state.literal_regex_candidates,
                        )
                    else:
                        valid_ids = _get_regex_tokens(
                            vocab, state.current_generated_text
                        )
                else:
                    # Literal strings still use prompt-derived candidates.
                    valid_ids = get_valid_tokens_for_value(
                        vocab, state.current_generated_text, "string",
                        state.treated_params,
                        len(state.current_function.parameters),
                        state.remaining_string_candidates
                    )
            else:
                valid_ids = token_validator(vocab,
                                            state.current_generated_text,
                                            ["true", "false"])
        else:
            candidates = get_candidat(state.current_state,
                                      state.current_function,
                                      state.treated_params)
            valid_ids = token_validator(vocab,
                                        state.current_generated_text,
                                        candidates)

        if not valid_ids:
            raise RuntimeError(
                "Constrained decoder has no valid token for "
                f"{state.current_state.name} ({state.current_param_name})"
            )
        masked_logits = logits_masker(valid_ids, logits)
        choosen_token_id = logit_chooser(masked_logits)
        state.current_generated_text += id_to_str[choosen_token_id]
        state.current_token_sequence.append(choosen_token_id)

        transitionner(state, prompt.prompt)

        if state.current_state == Step.PARAM_VALUE:
            print(f"  -> {state.current_state}"
                  f" [{state.current_param_name}]: "
                  f"{state.current_generated_text!r}")
        else:
            print(f"  -> {state.current_state}: "
                  f"{state.current_generated_text!r}")
        if is_generation_complete(state):
            break

    initial_len_final = initial_len
    valid_json = state.current_token_sequence[initial_len_final:]
    final_json_str = model.decode(valid_json)
    final_json = json.loads(final_json_str)
    final_json["prompt"] = prompt.prompt
    validated = OutputResult.model_validate(final_json)
    return validated.parameters
