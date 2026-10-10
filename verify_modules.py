import os
import json
import numpy as np
from src.inference.ai_detector_pipeline import AIDetector
from src.inference.grammar_pipeline import GrammarCorrector
from src.preprocessing.preprocess_ai_detection import clean_text_for_detection

def test_ai_detector():
    print("\n==========================================")
    print("TASK 1: VERIFY AI DETECTION MODEL")
    print("==========================================")
    
    detector = AIDetector()
    print("[1] Successfully loaded AIDetector from best_ai_detector.joblib")
    
    # Inspect Pipeline structure
    pipeline = detector.pipeline
    steps = pipeline.named_steps
    print(f"[2] Pipeline steps: {list(steps.keys())}")
    features = steps.get("features")
    clf = steps.get("clf")
    print(f"    - Feature union transformers: {[name for name, _ in features.transformer_list]}")
    print(f"    - Classifier: {clf.__class__.__name__} (params: C={getattr(clf, 'C', None)}, max_iter={getattr(clf, 'max_iter', None)})")
    
    # Check vocabulary / feature sizes
    word_vec = dict(features.transformer_list)["word_tfidf"]
    char_vec = dict(features.transformer_list)["char_tfidf"]
    print(f"    - Word TF-IDF vocab size: {len(word_vec.vocabulary_)}")
    print(f"    - Char TF-IDF vocab size: {len(char_vec.vocabulary_)}")
    
    # Test cases: Human-written vs AI-generated
    test_cases = [
        {
            "category": "Human (Informal/Conversational)",
            "text": "Honestly, I had no clue what was going on yesterday. We just went to the diner, ordered some waffles, and ended up chatting for like three hours about random stuff haha."
        },
        {
            "category": "Human (Academic/Journalistic)",
            "text": "The committee convened early on Tuesday morning to deliberate over the proposed fiscal reforms, ultimately postponing the vote after intense debate among senior delegates."
        },
        {
            "category": "Human (Short Query/Note)",
            "text": "Please make sure to bring your laptop and charger to tomorrow's meeting at 10 AM sharp."
        },
        {
            "category": "AI (ChatGPT Style Informational)",
            "text": "In conclusion, renewable energy sources play a crucial role in mitigating the impacts of climate change by reducing global greenhouse gas emissions and fostering sustainable economic growth."
        },
        {
            "category": "AI (GPT-3 Wikipedia Intro Style)",
            "text": "The Grand Central Terminal is a historic commuter rail terminal located at 42nd Street and Park Avenue in Midtown Manhattan, New York City. It serves commuters traveling on the Metro-North Railroad."
        },
        {
            "category": "AI (Technical Explanation)",
            "text": "Furthermore, gradient descent iteratively optimizes the objective function by calculating the partial derivatives with respect to each model parameter and updating the weights accordingly."
        }
    ]
    
    print("\n[3] Testing Manually Entered Human and AI Examples:")
    for idx, case in enumerate(test_cases, 1):
        raw_text = case["text"]
        # Test raw text prediction
        result_raw = detector.predict(raw_text)
        # Test preprocessed text prediction
        cleaned_text = clean_text_for_detection(raw_text)
        result_clean = detector.predict(cleaned_text)
        
        print(f"\n--- Case {idx}: [{case['category']}] ---")
        print(f"Input Text : {raw_text}")
        print(f"Prediction : {result_raw['label']} (AI Prob: {result_raw['ai_probability']:.4f}, Human Prob: {result_raw['human_probability']:.4f})")
        print(f"Confidence : {result_raw['confidence_score']:.4f} ({result_raw['confidence_level']}) | Assessment: {result_raw['assessment']}")
        print(f"Preproc Invariance: Matches Raw Pred = {result_raw['label'] == result_clean['label']} (Prob Diff: {abs(result_raw['ai_probability'] - result_clean['ai_probability']):.6f})")

    # Calibration & Confidence check
    print("\n[4] Confidence Score & Probability Calibration Analysis:")
    neutral_case = "The quick brown fox jumps over the lazy dog."
    neutral_res = detector.predict(neutral_case)
    print(f"    - Neutral sentence prediction: {neutral_res['label']}, AI Prob={neutral_res['ai_probability']}, Human Prob={neutral_res['human_probability']}")
    print(f"    - Logistic Regression predict_proba uses calibrated Platt sigmoid over decision function margins.")
    
def test_grammar_corrector():
    print("\n==========================================")
    print("TASK 2: DEBUG & VERIFY GRAMMAR CORRECTION")
    print("==========================================")
    
    corrector = GrammarCorrector()
    print(f"[1] Loaded GrammarCorrector with model: {corrector.model_name}")
    print(f"    - Device: {corrector.device}")
    print(f"    - Task prefix: '{corrector.task_prefix}'")
    
    # 10 Representative Test Sentences covering diverse error categories
    grammar_cases = [
        {
            "id": 1,
            "category": "Spelling and Typo Errors",
            "input": "She is a good freind and always halps me with my homwork."
        },
        {
            "id": 2,
            "category": "Subject-Verb Agreement",
            "input": "The list of items are on the desk in the office."
        },
        {
            "id": 3,
            "category": "Verb Tense Inconsistency",
            "input": "Yesterday, he go to the supermarket and buy fresh vegetables."
        },
        {
            "id": 4,
            "category": "Article Omission (a/an/the)",
            "input": "He is honest man who works as engineer in big company."
        },
        {
            "id": 5,
            "category": "Preposition Misuse",
            "input": "I am interested on learning machine learning since last year."
        },
        {
            "id": 6,
            "category": "Singular / Plural Noun Inflection",
            "input": "There are many person waiting outside with several luggage."
        },
        {
            "id": 7,
            "category": "Punctuation & Capitalization",
            "input": "where are you going i thought we had a meeting today"
        },
        {
            "id": 8,
            "category": "Already Correct Sentence (Clean)",
            "input": "Artificial intelligence algorithms require high-quality labeled data for supervised training."
        },
        {
            "id": 9,
            "category": "Already Correct Conversational",
            "input": "I really enjoy reading books in the library during my free time."
        },
        {
            "id": 10,
            "category": "Complex Multi-Error Sentence",
            "input": "Although she have been working here since two years, she do not knows all the policies."
        }
    ]
    
    print("\n[2] Testing 10 Representative Sentences:")
    for case in grammar_cases:
        res = corrector.correct_text(case["input"])
        print(f"\n--- Test #{case['id']}: [{case['category']}] ---")
        print(f"Original  : {case['input']}")
        print(f"Corrected : {res['corrected_text']}")
        print(f"Errors    : {res['total_errors']} edits detected (has_errors={res['has_errors']})")
        if res['sentence_corrections'] and res['sentence_corrections'][0]['edits']:
            for edit in res['sentence_corrections'][0]['edits']:
                print(f"  * Edit [{edit['action']}]: '{edit['original_tokens']}' -> '{edit['corrected_tokens']}'")

if __name__ == "__main__":
    test_ai_detector()
    test_grammar_corrector()
