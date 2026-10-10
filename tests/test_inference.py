import unittest
import os
from src.inference.ai_detector_pipeline import AIDetector
from src.inference.grammar_pipeline import GrammarCorrector
from src.inference.service import ContentAnalysisService

class TestInferencePipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = AIDetector()
        cls.corrector = GrammarCorrector()
        cls.service = ContentAnalysisService()

    def test_ai_detector_single_prediction(self):
        # Long passage (> 50 words)
        long_text = (
            "Deep learning architectures have revolutionized the field of computer vision and natural language processing. "
            "In recent years, transformer networks and large language models have achieved unprecedented performance across "
            "complex reasoning tasks, machine translation, and autonomous code generation. Researchers worldwide continue "
            "exploring novel scaling laws, instruction tuning paradigms, and alignment techniques to improve general reliability "
            "and factual coherence across multi-modal AI systems."
        )
        res = self.detector.predict(long_text)
        self.assertIn("is_ai", res)
        self.assertIn("label", res)
        self.assertIn("ai_probability", res)
        self.assertIn("human_probability", res)
        self.assertIn("confidence_score", res)
        self.assertIn("confidence_level", res)
        self.assertFalse(res["is_short_text"])
        self.assertIsNone(res["reliability_warning"])
        self.assertAlmostEqual(res["ai_probability"] + res["human_probability"], 1.0, places=2)

    def test_ai_detector_short_text_warning(self):
        # Short text (< 50 words)
        short_text = "The students gathered in the quiet library to study for their exams."
        res = self.detector.predict(short_text)
        self.assertTrue(res["is_short_text"])
        self.assertIsNotNone(res["reliability_warning"])
        self.assertEqual(res["confidence_level"], "Uncertain (Short Text)")
        self.assertEqual(res["word_count"], len(short_text.split()))

    def test_ai_detector_batch_prediction(self):
        texts = [
            "Hey what's up dude, you coming over tonight?",
            "Artificial intelligence systems are capable of generating human-like textual responses."
        ]
        results = self.detector.predict(texts)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results, list)

    def test_grammar_corrector_sentence(self):
        sentence = "She do not has enough money."
        res = self.corrector.correct_sentence(sentence)
        self.assertIn("original", res)
        self.assertIn("corrected", res)
        self.assertIn("has_errors", res)
        self.assertIn("edits", res)
        self.assertTrue(res["has_errors"])
        self.assertIn("does", res["corrected"].lower())

    def test_grammar_corrector_clean_sentence(self):
        sentence = "The students are studying in the library."
        res = self.corrector.correct_sentence(sentence)
        self.assertEqual(res["original"], res["corrected"])
        self.assertFalse(res["has_errors"])

    def test_content_analysis_service_unified(self):
        text = "Yesterday he go to the store. In conclusion, neural networks are effective tools."
        res = self.service.analyze_content(text)
        self.assertIn("input_text", res)
        self.assertIn("statistics", res)
        self.assertIn("ai_detection", res)
        self.assertIn("grammar_analysis", res)
        self.assertIn("execution_time_seconds", res)
        self.assertTrue(res["execution_time_seconds"] > 0)

if __name__ == "__main__":
    unittest.main()
