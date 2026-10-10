import os
import json
import re
import unicodedata
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_sentence(text: str) -> str:
    """
    Cleans and normalizes sentence text while preserving internal punctuation,
    capitalization, and subtle grammatical structure.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Strip known task instruction prefixes from CoEdit
    prefix_patterns = [
        r"^Remove all grammatical errors from this text:\s*",
        r"^Fix all the grammatical errors in the given text:\s*",
        r"^Fix the grammar errors in this sentence:\s*",
        r"^Correct grammatical errors:\s*",
        r"^Make this text fluent and grammatically correct:\s*"
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize excessive spaces while keeping single spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

def preprocess_grammar_data(
    raw_dir: str = "data/raw/grammar_correction",
    processed_dir: str = "data/processed/grammar_correction",
    reports_dir: str = "reports/dataset_quality",
    min_char_len: int = 10,
    max_char_len: int = 1000
) -> dict:
    """
    Cleans sentence pairs, validates alignment, removes corrupt records,
    and produces data quality reports.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    filtering_stats = {
        "coedit_raw_rows": 0,
        "jfleg_raw_rows": 0,
        "removed_empty_or_null": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_duplicate_pairs": 0,
        "identical_source_target_pairs": 0
    }

    # 1. Process Grammarly CoEdit GEC pairs
    coedit_records = []
    coedit_file = os.path.join(raw_dir, "raw_coedit_gec.jsonl")
    if os.path.exists(coedit_file):
        logger.info("Processing CoEdit GEC pairs...")
        raw_coedit_df = pd.read_json(coedit_file, orient="records", lines=True)
        filtering_stats["coedit_raw_rows"] = len(raw_coedit_df)

        for _, row in raw_coedit_df.iterrows():
            src_raw = row.get("src", "")
            tgt_raw = row.get("tgt", "")

            src_clean = clean_sentence(src_raw)
            tgt_clean = clean_sentence(tgt_raw)

            if not src_clean or not tgt_clean:
                filtering_stats["removed_empty_or_null"] += 1
                continue

            if len(src_clean) < min_char_len or len(tgt_clean) < min_char_len:
                filtering_stats["removed_too_short"] += 1
                continue

            if len(src_clean) > max_char_len or len(tgt_clean) > max_char_len:
                filtering_stats["removed_too_long"] += 1
                continue

            is_identical = (src_clean.lower() == tgt_clean.lower())
            if is_identical:
                filtering_stats["identical_source_target_pairs"] += 1

            coedit_records.append({
                "source_text": src_clean,
                "target_text": tgt_clean,
                "is_changed": not is_identical,
                "dataset_source": "coedit_gec"
            })

    coedit_df = pd.DataFrame(coedit_records)
    pre_dedup = len(coedit_df)
    coedit_df = coedit_df.drop_duplicates(subset=["source_text", "target_text"]).reset_index(drop=True)
    filtering_stats["removed_duplicate_pairs"] = pre_dedup - len(coedit_df)

    # 2. Process JFLEG benchmark evaluation set
    jfleg_records = []
    jfleg_file = os.path.join(raw_dir, "raw_jfleg_test.jsonl")
    if os.path.exists(jfleg_file):
        logger.info("Processing JFLEG benchmark...")
        raw_jfleg_df = pd.read_json(jfleg_file, orient="records", lines=True)
        filtering_stats["jfleg_raw_rows"] = len(raw_jfleg_df)

        for _, row in raw_jfleg_df.iterrows():
            src_raw = row.get("sentence", "")
            corrections = row.get("corrections", [])

            src_clean = clean_sentence(src_raw)
            if not src_clean:
                continue

            # Clean all reference corrections
            cleaned_refs = [clean_sentence(c) for c in corrections if clean_sentence(c)]
            if cleaned_refs:
                jfleg_records.append({
                    "source_text": src_clean,
                    "target_text": cleaned_refs[0], # Primary reference
                    "reference_corrections": cleaned_refs,
                    "dataset_source": "jfleg_benchmark"
                })

    jfleg_df = pd.DataFrame(jfleg_records)

    # Save processed datasets
    coedit_processed_path = os.path.join(processed_dir, "processed_coedit_gec.parquet")
    coedit_df.to_parquet(coedit_processed_path, index=False)
    coedit_csv_path = os.path.join(processed_dir, "processed_coedit_gec.csv")
    coedit_df.to_csv(coedit_csv_path, index=False)

    jfleg_processed_path = os.path.join(processed_dir, "processed_jfleg_test.parquet")
    jfleg_df.to_parquet(jfleg_processed_path, index=False)
    jfleg_csv_path = os.path.join(processed_dir, "processed_jfleg_test.csv")
    jfleg_df.to_csv(jfleg_csv_path, index=False)

    logger.info(f"Saved processed CoEdit ({len(coedit_df)} pairs) and JFLEG ({len(jfleg_df)} benchmark pairs)")

    # Data Quality Report
    coedit_df["src_char_len"] = coedit_df["source_text"].str.len()
    coedit_df["tgt_char_len"] = coedit_df["target_text"].str.len()
    coedit_df["src_word_count"] = coedit_df["source_text"].apply(lambda t: len(t.split()))

    quality_report = {
        "module": "Grammar Error Detection and Correction",
        "coedit_final_pairs": len(coedit_df),
        "jfleg_benchmark_pairs": len(jfleg_df),
        "filtering_statistics": filtering_stats,
        "coedit_changed_ratio": float(coedit_df["is_changed"].mean()) if "is_changed" in coedit_df else 1.0,
        "text_statistics": {
            "source_char_length_mean": float(coedit_df["src_char_len"].mean()),
            "source_char_length_std": float(coedit_df["src_char_len"].std()),
            "target_char_length_mean": float(coedit_df["tgt_char_len"].mean()),
            "source_word_count_mean": float(coedit_df["src_word_count"].mean())
        }
    }

    report_path = os.path.join(reports_dir, "grammar_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved grammar data quality report to {report_path}")

    return {"coedit": coedit_df, "jfleg": jfleg_df}

if __name__ == "__main__":
    preprocess_grammar_data()
