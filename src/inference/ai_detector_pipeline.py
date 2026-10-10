import os
import joblib
from typing import List, Union, Dict, Any

class AIDetector:
    """
    High-level production inference pipeline for AI vs Human text detection.
    """
    def __init__(
        self,
        model_path: str = "models/ai_detection/best_ai_detector.joblib",
        min_reliable_words: int = 50
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
        self.pipeline = joblib.load(model_path)
        self.classes_ = ["human", "ai"]
        self.min_reliable_words = min_reliable_words

    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        preds = self.pipeline.predict(texts)
        probs = self.pipeline.predict_proba(texts)

        results = []
        for t, pred, prob in zip(texts, preds, probs):
            human_prob = float(prob[0])
            ai_prob = float(prob[1])
            is_ai = bool(pred == 1)
            raw_confidence = ai_prob if is_ai else human_prob

            words = t.strip().split()
            word_count = len(words)
            is_short = word_count < self.min_reliable_words

            raw_assessment = "Likely AI-Generated" if is_ai else "Likely Human-Written"

            if is_short:
                confidence_level = "Uncertain (Short Text)"
                reliability_warning = (
                    f"Input contains only {word_count} words (minimum recommended: {self.min_reliable_words} words). "
                    "Statistical n-gram detection on short sentences has high variance and should not be considered definitive."
                )
                assessment = f"{raw_assessment} (Unreliable / Short Sample)"
            else:
                reliability_warning = None
                if raw_confidence >= 0.85:
                    confidence_level = "High"
                elif raw_confidence >= 0.65:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low / Uncertain"
                assessment = raw_assessment

            results.append({
                "text_snippet": t[:100] + ("..." if len(t) > 100 else ""),
                "is_ai": is_ai,
                "label": "AI-Generated" if is_ai else "Human-Written",
                "ai_probability": round(ai_prob, 4),
                "human_probability": round(human_prob, 4),
                "confidence_score": round(raw_confidence, 4),
                "confidence_level": confidence_level,
                "assessment": assessment,
                "word_count": word_count,
                "min_reliable_words": self.min_reliable_words,
                "is_short_text": is_short,
                "reliability_warning": reliability_warning
            })

        return results[0] if is_single else results

if __name__ == "__main__":
    detector = AIDetector()
    samples = [
        "In artificial intelligence, deep learning models utilize hierarchical feature representations.",
        "Yo yesterday me and my bro went to the park and grabbed some ice cream it was super fun haha"
    ]
    for sample in samples:
        print(detector.predict(sample))
