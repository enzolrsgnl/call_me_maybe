import math
from enum import Enum, auto
from .models import Function, EntryPrompt, OutputResult
from llm_sdk import Small_LLM_Model
import json


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
    for index, logit in enumerate(logits):
        if index not in valids_token:
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
                 treated_params: list[str]) -> None:
        self.current_state = current_state
        self.current_function = current_function
        self.current_param_name = current_param_name
        self.current_generated_text = current_generated_text
        self.current_token_sequence = current_token_sequence
        self.treated_params = treated_params


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


def get_valid_tokens_for_value(vocab: dict[str, int], already_generated: str,
                               param_type: str, treated_params: list[str],
                               parameters_number: int) -> list[int]:
    """ This function is a token validator for the param_value state,
    when we are treating the parameter value,
    we have to validate the token sequence cause the generation is free """
    valids_token_id = []
    for token_string, token_id in vocab.items():
        if param_type == "number":
            number_validation = is_valid_number_token(token_string,
                                                      already_generated)
            if len(treated_params) + 1 >= parameters_number:
                closing_char = "}"
            else:
                closing_char = ","
            if token_string == closing_char:
                valids_token_id.append(token_id)
            if number_validation:
                valids_token_id.append(token_id)
        elif param_type == "string":
            string_validation = is_valid_string_token(token_string,
                                                      already_generated)
            if string_validation:
                valids_token_id.append(token_id)
    return valids_token_id


def transitionner(state: GenerationState) -> None:
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
            if (state.current_generated_text[-1] == '"'
               and len(state.current_generated_text) > 1):
                state.treated_params.append(state.current_param_name)
                state.current_state = get_next_step(state.current_state,
                                                    state.current_function,
                                                    state.treated_params)
                state.current_generated_text = ""
        elif param_type == "number":
            last_char = state.current_generated_text[-1]
            if last_char not in "0123456789.-":
                state.current_generated_text = (
                    state.current_generated_text[:-1]
                )
                state.treated_params.append(state.current_param_name)
                state.current_state = get_next_step(state.current_state,
                                                    state.current_function,
                                                    state.treated_params)
                state.current_generated_text = last_char
                transitionner(state)
        elif param_type == "boolean":
            remaining = filter_candidate(["true", "false"],
                                         state.current_generated_text)
            if (remaining[0] == state.current_generated_text
               and len(remaining) == 1):
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
    enriched_prompt = (
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {function.parameters}\n"
        f"Question: {prompt.prompt}"
    )

    encoded = model.encode(enriched_prompt)
    first_batch = encoded[0]
    initial_tokens = first_batch.tolist()
    initial_len = len(initial_tokens)
    state = (
        GenerationState(Step.OPEN_BRACE, function, "", "", initial_tokens, [])
        )
    id_to_str = {v: k for k, v in vocab.items()}
    while (state.current_state != Step.CLOSING_BRACE
           or state.current_generated_text != step_target[Step.CLOSING_BRACE]):
        logits = model.get_logits_from_input_ids(state.current_token_sequence)
        parameters_effectif = len(state.current_function.parameters)
        if state.current_state == Step.PARAM_VALUE:
            current_param = (
                state.current_function.parameters[state.current_param_name]
                )
            param_type = current_param.type
            if param_type in ["string", "number"]:
                valid_ids = (
                    get_valid_tokens_for_value(vocab,
                                               state.current_generated_text,
                                               param_type,
                                               state.treated_params,
                                               parameters_effectif)
                    )
            else:
                valid_ids = (
                    token_validator(vocab, state.current_generated_text,
                                    ["true", "false"])
                    )
        else:
            candidates = (
                get_candidat(state.current_state,
                             state.current_function,
                             state.treated_params)
                )
            valid_ids = (
                token_validator(vocab,
                                state.current_generated_text,
                                candidates)
                )

        masked_logits = logits_masker(valid_ids, logits)
        choosen_token_id = logit_chooser(masked_logits)
        state.current_generated_text = (
            state.current_generated_text
            + id_to_str[choosen_token_id]
            )
        state.current_token_sequence.append(choosen_token_id)
        transitionner(state)
        print(f"  -> {state.current_state}: {state.current_generated_text!r}")
        if is_generation_complete(state):
            break

    valid_json = state.current_token_sequence[initial_len:]
    final_json_str = model.decode(valid_json)
    final_json = json.loads(final_json_str)

    final_json["prompt"] = prompt.prompt
    validated = OutputResult.model_validate(final_json)
    return validated.parameters
