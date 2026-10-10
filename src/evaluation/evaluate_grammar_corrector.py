import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_grammar_test_sets(
    model_metadata_path: str = "models/grammar_correction/grammar_model_metadata.json",
    splits_dir: str = "data/splits/grammar_correction",
    reports_dir: str = "reports",
    max_eval_samples: int = 150
) -> dict:
    """
    Evaluates the Grammar Error Correction model on both the held-out CoEdit test split
    and the independent JFLEG benchmark corpus.
    """
    eval_dir = os.path.join(reports_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(model_metadata_path):
        model_name = "vennify/t5-base-grammar-correction"
        task_prefix = ""
    else:
        with open(model_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_name = meta.get("selected_pretrained_model", "vennify/t5-base-grammar-correction")
        task_prefix = meta.get("task_prefix", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Grammar Model '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    def run_inference(sentences):
        preds = []
        for s in sentences:
            prompt = f"{task_prefix}{s}" if task_prefix else s
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            preds.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return preds

    # 1. Evaluate on CoEdit Test Split
    coedit_test_file = os.path.join(splits_dir, "test.parquet")
    if os.path.exists(coedit_test_file):
        coedit_test_df = pd.read_parquet(coedit_test_file)
    else:
        coedit_test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    coedit_sample = coedit_test_df.head(max_eval_samples)
    coedit_srcs = coedit_sample["source_text"].tolist()
    coedit_refs = [[t] for t in coedit_sample["target_text"].tolist()]
    coedit_preds = run_inference(coedit_srcs)

    coedit_bleu = sacrebleu.corpus_bleu(coedit_preds, coedit_refs).score
    coedit_chrf = sacrebleu.corpus_chrf(coedit_preds, coedit_refs).score
    coedit_em = sum(1 for p, r in zip(coedit_preds, coedit_sample["target_text"]) if p.lower() == r.lower())
    coedit_em_pct = (coedit_em / len(coedit_preds)) * 100

    # 2. Evaluate on JFLEG Benchmark
    jfleg_test_file = os.path.join(splits_dir, "jfleg_benchmark_test.parquet")
    if os.path.exists(jfleg_test_file):
        jfleg_df = pd.read_parquet(jfleg_test_file)
    else:
        jfleg_df = pd.read_csv(os.path.join(splits_dir, "jfleg_benchmark_test.csv"))

    jfleg_sample = jfleg_df.head(max_eval_samples)
    jfleg_srcs = jfleg_sample["source_text"].tolist()
    
    # Collect multi-reference corrections if available
    raw_refs = jfleg_sample["reference_corrections"].iloc[0] if "reference_corrections" in jfleg_sample.columns else None
    if raw_refs is not None:
        parsed_refs = []
        for refs in jfleg_sample["reference_corrections"]:
            if isinstance(refs, str):
                try:
                    refs = json.loads(refs)
                except Exception:
                    refs = [refs]
            elif isinstance(refs, (list, np.ndarray)):
                refs = list(refs)
            else:
                refs = [str(refs)]
            parsed_refs.append(refs)

        jfleg_refs_transposed = []
        max_refs = max(len(r) for r in parsed_refs)
        for r_idx in range(max_refs):
            ref_list = []
            for r in parsed_refs:
                ref_list.append(r[r_idx] if r_idx < len(r) else r[0])
            jfleg_refs_transposed.append(ref_list)
    else:
        jfleg_refs_transposed = [[t] for t in jfleg_sample["target_text"].tolist()]

    jfleg_preds = run_inference(jfleg_srcs)
    jfleg_bleu = sacrebleu.corpus_bleu(jfleg_preds, jfleg_refs_transposed).score
    jfleg_chrf = sacrebleu.corpus_chrf(jfleg_preds, jfleg_refs_transposed).score

    # Collect qualitative examples: successful vs challenging
    successful_examples = []
    challenging_examples = []

    for i in range(len(coedit_srcs)):
        src = coedit_srcs[i]
        tgt = coedit_refs[0][i] if isinstance(coedit_refs[0], list) and len(coedit_refs) == 1 else coedit_refs[i][0]
        pred = coedit_preds[i]

        item = {
            "source_raw": src,
            "reference_target": tgt,
            "model_corrected": pred
        }

        if pred.lower() == tgt.lower():
            if len(successful_examples) < 6:
                successful_examples.append(item)
        else:
            if len(challenging_examples) < 6:
                challenging_examples.append(item)

    report = {
        "module": "Module B: Grammar Error Detection and Correction",
        "model_name": model_name,
        "coedit_test_evaluation": {
            "test_samples_evaluated": len(coedit_srcs),
            "sacrebleu_score": float(coedit_bleu),
            "chrf_plus_score": float(coedit_chrf),
            "exact_match_percentage": float(coedit_em_pct)
        },
        "jfleg_benchmark_evaluation": {
            "benchmark_samples_evaluated": len(jfleg_srcs),
            "multi_ref_bleu_score": float(jfleg_bleu),
            "multi_ref_chrf_score": float(jfleg_chrf)
        },
        "successful_correction_examples": successful_examples,
        "challenging_correction_examples": challenging_examples,
        "error_categories_addressed": [
            "Subject-Verb Agreement (e.g., 'he go' -> 'he goes')",
            "Verb Tense Inconsistencies (e.g., 'yesterday I go' -> 'yesterday I went')",
            "Singular/Plural Noun Inflections",
            "Preposition and Article Omission/Misuse (e.g., 'interested on' -> 'interested in')",
            "Spelling and Typographical Normalization",
            "Punctuation and Capitalization Alignment"
        ],
        "limitations": [
            "Severe colloquial slangs or heavy ungrammatical jargon may be over-regularized.",
            "Multi-sentence inputs should be sentence-tokenized for optimal precision."
        ]
    }

    report_path = os.path.join(eval_dir, "grammar_test_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved comprehensive Grammar Test Evaluation Report to {report_path}")
    logger.info(f"CoEdit BLEU: {coedit_bleu:.2f}, chrF: {coedit_chrf:.2f}, Exact Match: {coedit_em_pct:.1f}%")
    logger.info(f"JFLEG Multi-Ref BLEU: {jfleg_bleu:.2f}, chrF: {jfleg_chrf:.2f}")

    return report

if __name__ == "__main__":
    evaluate_grammar_test_sets()
