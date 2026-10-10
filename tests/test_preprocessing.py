import unittest
from src.preprocessing.preprocess_ai_detection import clean_text_for_detection
from src.preprocessing.preprocess_grammar import clean_sentence

class TestPreprocessing(unittest.TestCase):
    def test_clean_text_ai_detection(self):
        sample = "  This is   a \t text \n\n\n with   whitespace.  "
        cleaned = clean_text_for_detection(sample)
        self.assertEqual(cleaned, "This is a text\n\nwith whitespace.")

    def test_clean_text_null_handling(self):
        self.assertEqual(clean_text_for_detection(None), "")
        self.assertEqual(clean_text_for_detection(123), "")

    def test_clean_sentence_grammar_prefix_removal(self):
        sample = "Remove all grammatical errors from this text: He are going to home."
        cleaned = clean_sentence(sample)
        self.assertEqual(cleaned, "He are going to home.")

    def test_clean_sentence_punctuation_preservation(self):
        sample = "Don't break user's commas, apostrophes; or dashes!"
        cleaned = clean_sentence(sample)
        self.assertEqual(cleaned, "Don't break user's commas, apostrophes; or dashes!")

if __name__ == "__main__":
    unittest.main()
