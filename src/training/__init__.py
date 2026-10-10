# Model Training Module
from .train_ai_detector import train_ai_detection_models
from .train_grammar_corrector import setup_and_evaluate_grammar_model

__all__ = ["train_ai_detection_models", "setup_and_evaluate_grammar_model"]
