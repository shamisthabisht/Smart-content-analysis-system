import os
import time
from typing import Dict, Any, List, Optional
from .ai_detector_pipeline import AIDetector
from .grammar_pipeline import GrammarCorrector

class ContentAnalysisService:
    """
    Unified high-level inference service for AI Content Analysis System.
    Integrates:
      1. Module A: AI-Generated vs Human-Written Text Detection
      2. Module B: Grammar Error Detection and Correction
    """
    def __init__(
        self,
        ai_detector_path: str = "models/ai_detection/best_ai_detector.joblib",
        grammar_metadata_path: str = "models/grammar_correction/grammar_model_metadata.json",
        device: Optional[str] = None
    ):
        self.ai_detector = AIDetector(model_path=ai_detector_path)
        self.grammar_corrector = GrammarCorrector(metadata_path=grammar_metadata_path, device=device)

    def detect_ai(self, text: str) -> Dict[str, Any]:
        """Runs AI detection on the input text."""
        return self.ai_detector.predict(text)

    def correct_grammar(self, text: str) -> Dict[str, Any]:
        """Runs grammar correction and returns edits."""
        return self.grammar_corrector.correct_text(text)

    def analyze_content(self, text: str, options: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Runs comprehensive analysis (AI detection + Grammar correction) on input text.
        
        Args:
            text: Input string to analyze.
            options: Optional dict flags {"enable_ai_detection": True, "enable_grammar": True}
        
        Returns:
            Structured dictionary containing AI detection results, grammar edits,
            readability stats, and latency telemetry.
        """
        if options is None:
            options = {"enable_ai_detection": True, "enable_grammar": True}

        start_time = time.time()
        
        # 1. Text statistics
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        results: Dict[str, Any] = {
            "input_text": text,
            "statistics": {
                "char_count": char_count,
                "word_count": word_count,
                "sentence_count": len([s for s in text.split(".") if s.strip()])
            },
            "ai_detection": None,
            "grammar_analysis": None,
            "execution_time_seconds": 0.0
        }

        # 2. Module A: AI Detection
        if options.get("enable_ai_detection", True):
            ai_res = self.ai_detector.predict(text)
            results["ai_detection"] = ai_res

        # 3. Module B: Grammar Correction
        if options.get("enable_grammar", True):
            grammar_res = self.grammar_corrector.correct_text(text)
            results["grammar_analysis"] = grammar_res

        results["execution_time_seconds"] = round(time.time() - start_time, 4)
        return results

if __name__ == "__main__":
    service = ContentAnalysisService()
    sample = "In this research paper, we demonstrates novel methodology for deep neural network training. It produce very high accuracy."
    analysis = service.analyze_content(sample)
    import json
    print(json.dumps(analysis, indent=2))
