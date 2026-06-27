from .models import EntryPrompt, Function
from .generator import token_validator, logits_masker
from .generator import logit_chooser, get_function_name, filter_candidate
from llm_sdk import Small_LLM_Model


def func_chooser(vocab: dict[str, int], model: Small_LLM_Model,
                 prompt: EntryPrompt, functions: list[Function]) -> Function:
    """ This function choose the final function to return """
    current_generated_text = ""
    descriptions = ""
    for f in functions:
        descriptions = descriptions + f"- {f.name}: {f.description}\n"
    enriched_prompt = (
        f"Available functions:\n{descriptions}\n"
        f"Question: {prompt.prompt}\n"
        f"Answer with a single function name only:\n"
    )
    token_sequence = model.encode(enriched_prompt)[0].tolist()
    candidates = get_function_name(functions)
    id_to_str = {v: k for k, v in vocab.items()}
    while len(candidates) != 1 or candidates[0] != current_generated_text:
        logits = model.get_logits_from_input_ids(token_sequence)
        valids_tokens = token_validator(vocab, current_generated_text,
                                        candidates)
        logits_masked = logits_masker(valids_tokens, logits)
        token_id_choosen = logit_chooser(logits_masked)
        current_generated_text = (
            current_generated_text + id_to_str[token_id_choosen]
            )
        token_sequence.append(token_id_choosen)
        remaining_candidates = (
            filter_candidate(candidates, current_generated_text)
            )
        candidates = remaining_candidates

    for f in functions:
        if f.name == candidates[0]:
            return f

    raise ValueError(f"Non matching function for: {candidates[0]}")
