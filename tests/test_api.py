import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize TestClient which runs lifespan startup
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["ai_detector_loaded"])
        self.assertTrue(data["grammar_model_loaded"])
        self.assertIn("version", data)

    def test_ai_detection_endpoint_valid(self):
        payload = {
            "text": "Deep learning architectures have revolutionized the field of computer vision and natural language processing."
        }
        response = self.client.post("/api/analyze/ai-detection", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_ai", data)
        self.assertIn("label", data)
        self.assertIn("ai_probability", data)
        self.assertIn("human_probability", data)
        self.assertIn("confidence_score", data)
        self.assertIn("confidence_level", data)
        self.assertIn("assessment", data)
        self.assertIn("is_short_text", data)
        self.assertIn("word_count", data)
        self.assertTrue(data["execution_time_seconds"] >= 0)

    def test_ai_detection_endpoint_empty_error(self):
        payload = {"text": "   "}
        response = self.client.post("/api/analyze/ai-detection", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_grammar_endpoint_valid(self):
        payload = {
            "text": "She go to school yesterday and she have many book."
        }
        response = self.client.post("/api/analyze/grammar", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("original_text", data)
        self.assertIn("corrected_text", data)
        self.assertTrue(data["has_errors"])
        self.assertTrue(data["total_errors"] > 0)
        self.assertIsInstance(data["sentence_corrections"], list)
        self.assertIn("went", data["corrected_text"])

    def test_grammar_endpoint_clean_text(self):
        payload = {
            "text": "The students are studying in the library."
        }
        response = self.client.post("/api/analyze/grammar", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["has_errors"])
        self.assertEqual(data["total_errors"], 0)

    def test_combined_endpoint(self):
        payload = {
            "text": "In this article we shows how machine learning works.",
            "enable_ai_detection": True,
            "enable_grammar": True
        }
        response = self.client.post("/api/analyze/combined", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("statistics", data)
        self.assertIsNotNone(data["ai_detection"])
        self.assertIsNotNone(data["grammar_analysis"])
        self.assertTrue(data["execution_time_seconds"] > 0)

if __name__ == "__main__":
    unittest.main()
