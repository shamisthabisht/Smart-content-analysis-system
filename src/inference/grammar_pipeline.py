import os
import json
import difflib
import re
import torch
from typing import List, Union, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GrammarCorrector:
    """
    Production-ready inference pipeline for Grammar Error Detection & Correction.
    Generates fully corrected text and provides word-level edit diffs.
    """
    def __init__(
        self,
        metadata_path: str = "models/grammar_correction/grammar_model_metadata.json",
        device: str = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.model_name = meta.get("selected_pretrained_model", "vennify/t5-base-grammar-correction")
            self.task_prefix = meta.get("task_prefix", "")
        else:
            self.model_name = "vennify/t5-base-grammar-correction"
            self.task_prefix = ""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits paragraph into sentences while preserving order."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_word_diffs(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """Computes word-level diffs between original and corrected sentences."""
        orig_words = original.split()
        corr_words = corrected.split()
        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)

        diffs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            orig_segment = " ".join(orig_words[i1:i2])
            corr_segment = " ".join(corr_words[j1:j2])
            if tag != "equal":
                diffs.append({
                    "action": tag,  # 'replace', 'delete', 'insert'
                    "original_tokens": orig_segment,
                    "corrected_tokens": corr_segment
                })
        return diffs

    def correct_sentence(self, sentence: str) -> Dict[str, Any]:
        prompt = f"{self.task_prefix}{sentence}" if self.task_prefix else sentence
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        diffs = self._compute_word_diffs(sentence, corrected)
        has_errors = len(diffs) > 0

        return {
            "original": sentence,
            "corrected": corrected,
            "has_errors": has_errors,
            "error_count": len(diffs),
            "edits": diffs
        }

    def correct_text(self, text: str) -> Dict[str, Any]:
        """
        Corrects multi-sentence text or paragraphs.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return {
                "original_text": text,
                "corrected_text": text,
                "has_errors": False,
                "total_errors": 0,
                "sentence_corrections": []
            }

        sentence_results = [self.correct_sentence(s) for s in sentences]
        corrected_full = " ".join([res["corrected"] for res in sentence_results])
        total_errors = sum(res["error_count"] for res in sentence_results)
        has_errors = total_errors > 0

        return {
            "original_text": text,
            "corrected_text": corrected_full,
            "has_errors": has_errors,
            "total_errors": total_errors,
            "sentence_corrections": sentence_results
        }

if __name__ == "__main__":
    corrector = GrammarCorrector()
    test_sample = "She go to school yesterday and she have many book."
    res = corrector.correct_text(test_sample)
    print(json.dumps(res, indent=2))
