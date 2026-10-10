import os
import json
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_and_evaluate_grammar_model(
    model_name: str = "vennify/t5-base-grammar-correction",
    fallback_model: str = "google-t5/t5-small",
    splits_dir: str = "data/splits/grammar_correction",
    models_dir: str = "models/grammar_correction",
    reports_dir: str = "reports",
    eval_sample_limit: int = 100
) -> dict:
    """
    Sets up the Seq2Seq Grammar Correction model and tokenizer,
    evaluates on validation samples, and saves configuration & metadata.
    """
    os.makedirs(models_dir, exist_ok=True)
    eval_reports_dir = os.path.join(reports_dir, "evaluation")
    os.makedirs(eval_reports_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for Grammar Correction Model.")

    logger.info(f"Attempting to load model '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        selected_model_name = model_name
    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}. Falling back to {fallback_model}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
        selected_model_name = fallback_model

    model.to(device)
    model.eval()

    # Save tokenizer and local model config
    tokenizer_dir = os.path.join(models_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    
    config_metadata = {
        "model_architecture": "T5 Seq2Seq Text-to-Text Grammar Corrector",
        "selected_pretrained_model": selected_model_name,
        "device": device,
        "max_length": 128,
        "num_beams": 4,
        "tokenizer_path": tokenizer_dir,
        "task_prefix": "grammar: " if "t5-small" in selected_model_name else ""
    }

    config_path = os.path.join(models_dir, "grammar_model_metadata.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_metadata, f, indent=2)

    logger.info(f"Grammar model setup complete. Saved metadata to {config_path}")

    # Evaluate on Validation split
    val_file = os.path.join(splits_dir, "val.parquet")
    if os.path.exists(val_file):
        val_df = pd.read_parquet(val_file)
    else:
        val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))

    eval_subset = val_df.head(eval_sample_limit)
    sources = eval_subset["source_text"].tolist()
    references = [[t] for t in eval_subset["target_text"].tolist()]
    predictions = []

    logger.info(f"Running validation evaluation on {len(sources)} sample pairs...")
    for text in sources:
        prompt = f"grammar: {text}" if "t5-small" in selected_model_name else text
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(pred_text)

    # Compute BLEU / GLEU metric using sacrebleu
    bleu_score = sacrebleu.corpus_bleu(predictions, references).score
    chrf_score = sacrebleu.corpus_chrf(predictions, references).score
    exact_matches = sum(1 for p, r in zip(predictions, eval_subset["target_text"]) if p.strip().lower() == r.strip().lower())
    exact_match_acc = (exact_matches / len(predictions)) * 100

    logger.info(f"Validation Evaluation -> BLEU: {bleu_score:.2f}, chrF++: {chrf_score:.2f}, Exact Match: {exact_match_acc:.1f}%")

    # Sample comparison records
    sample_comparisons = []
    for i in range(min(5, len(sources))):
        sample_comparisons.append({
            "original_incorrect": sources[i],
            "reference_target": references[i][0],
            "model_prediction": predictions[i]
        })

    val_report = {
        "model_name": selected_model_name,
        "validation_samples_evaluated": len(sources),
        "metrics": {
            "sacrebleu_score": float(bleu_score),
            "chrf_score": float(chrf_score),
            "exact_match_percentage": float(exact_match_acc)
        },
        "sample_comparisons": sample_comparisons
    }

    val_report_path = os.path.join(eval_reports_dir, "grammar_validation_report.json")
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    logger.info(f"Saved grammar validation report to {val_report_path}")

    return {
        "model_name": selected_model_name,
        "metrics": val_report["metrics"],
        "sample_comparisons": sample_comparisons
    }

if __name__ == "__main__":
    setup_and_evaluate_grammar_model()
