import unittest

from src.generator import (
    extract_quoted_phrases,
    extract_literal_regex_candidates,
    is_regex_parameter,
    is_valid_regex_token,
)
from src.models import Function


class RegexConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.function = Function.model_validate({
            "name": "substitute",
            "description": "Replace matches of a regex pattern.",
            "parameters": {
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "returns": {"type": "string"},
        })

    def test_only_regex_slot_gets_regex_grammar(self) -> None:
        self.assertTrue(is_regex_parameter("regex", self.function))
        self.assertFalse(is_regex_parameter("source_string", self.function))
        self.assertFalse(is_regex_parameter("replacement", self.function))

    def test_literal_regex_for_cat_substitution(self) -> None:
        self.assertTrue(is_valid_regex_token('"cat"', ""))

    def test_vowel_character_class(self) -> None:
        self.assertTrue(is_valid_regex_token('"[aeiouAEIOU]"', ""))

    def test_digit_character_class(self) -> None:
        self.assertTrue(is_valid_regex_token('"[0-9]"', ""))

    def test_rejects_drifting_alternation_and_quantifiers(self) -> None:
        self.assertFalse(is_valid_regex_token('".{2,}+a|e|i|o|u"', ""))

    def test_rejects_overly_broad_wildcard(self) -> None:
        self.assertFalse(is_valid_regex_token('"."', ""))

    def test_extracts_double_quoted_source_with_apostrophe(self) -> None:
        prompt = 'Replace numbers in "Hello 34 I\'m 233 years old"'
        self.assertEqual(
            extract_quoted_phrases(prompt),
            ['"Hello 34 I\'m 233 years old"'],
        )

    def test_extracts_literal_word_as_regex_candidate(self) -> None:
        prompt = "Substitute the word 'cat' with 'dog' in 'a cat'"
        self.assertEqual(extract_literal_regex_candidates(prompt), ['"cat"'])

    def test_category_is_not_mistaken_for_literal_target(self) -> None:
        prompt = "Replace all vowels in 'Programming' with asterisks"
        self.assertEqual(extract_literal_regex_candidates(prompt), [])


if __name__ == "__main__":
    unittest.main()
