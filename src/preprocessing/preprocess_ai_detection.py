import os
import json
import re
import unicodedata
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_text_for_detection(text: str) -> str:
    """
    Cleans text while preserving essential stylistic markers, punctuation,
    capitalization, and paragraph flow.
    """
    if not isinstance(text, str):
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Clean whitespace per line and reconstruct paragraphs
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    paragraphs = []
    current_para = []
    for line in lines:
        if line:
            current_para.append(line)
        elif current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))
    return "\n\n".join(paragraphs).strip()

def preprocess_ai_detection_data(
    raw_dir: str = "data/raw/ai_detection",
    processed_dir: str = "data/processed/ai_detection",
    reports_dir: str = "reports/dataset_quality",
    min_char_len: int = 50,
    max_char_len: int = 5000,
    target_samples_per_source: int = 5000
) -> pd.DataFrame:
    """
    Harmonizes schemas, cleans text, deduplicates, and generates data quality reports.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    records = []
    filtering_stats = {
        "gpt_wiki_intro_raw_rows": 0,
        "hc3_raw_rows": 0,
        "removed_empty_or_null": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_exact_duplicates": 0
    }

    # 1. Process GPT-wiki-intro
    wiki_raw_file = os.path.join(raw_dir, "raw_gpt_wiki_intro.jsonl")
    if os.path.exists(wiki_raw_file):
        logger.info("Processing GPT-wiki-intro...")
        wiki_df = pd.read_json(wiki_raw_file, orient="records", lines=True)
        filtering_stats["gpt_wiki_intro_raw_rows"] = len(wiki_df)

        count = 0
        for _, row in wiki_df.iterrows():
            if count >= target_samples_per_source:
                break
            human_text = clean_text_for_detection(row.get("wiki_intro", ""))
            ai_text = clean_text_for_detection(row.get("generated_intro", ""))

            if human_text:
                records.append({
                    "text": human_text,
                    "label": 0, # 0 = Human-written
                    "label_name": "human",
                    "source_dataset": "gpt_wiki_intro",
                    "generator": "human",
                    "domain": "encyclopedic_wikipedia"
                })
            if ai_text:
                records.append({
                    "text": ai_text,
                    "label": 1, # 1 = AI-generated
                    "label_name": "ai",
                    "source_dataset": "gpt_wiki_intro",
                    "generator": "gpt3",
                    "domain": "encyclopedic_wikipedia"
                })
            count += 1

    # 2. Process HC3
    hc3_raw_file = os.path.join(raw_dir, "raw_hc3.jsonl")
    if os.path.exists(hc3_raw_file):
        logger.info("Processing HC3...")
        hc3_df = pd.read_json(hc3_raw_file, orient="records", lines=True)
        filtering_stats["hc3_raw_rows"] = len(hc3_df)

        count = 0
        for _, row in hc3_df.iterrows():
            if count >= target_samples_per_source:
                break
            human_answers = row.get("human_answers", [])
            chatgpt_answers = row.get("chatgpt_answers", [])

            if isinstance(human_answers, list) and len(human_answers) > 0:
                h_text = clean_text_for_detection(human_answers[0])
                if h_text:
                    records.append({
                        "text": h_text,
                        "label": 0,
                        "label_name": "human",
                        "source_dataset": "hc3",
                        "generator": "human",
                        "domain": "multi_domain_qa"
                    })

            if isinstance(chatgpt_answers, list) and len(chatgpt_answers) > 0:
                ai_text = clean_text_for_detection(chatgpt_answers[0])
                if ai_text:
                    records.append({
                        "text": ai_text,
                        "label": 1,
                        "label_name": "ai",
                        "source_dataset": "hc3",
                        "generator": "chatgpt",
                        "domain": "multi_domain_qa"
                    })
            count += 1

    df = pd.DataFrame(records)
    original_count = len(df)
    logger.info(f"Total harmonized raw records extracted: {original_count}")

    # Filtering step 1: Empty or null text
    non_empty = df["text"].notnull() & (df["text"].str.strip() != "")
    filtering_stats["removed_empty_or_null"] = int((~non_empty).sum())
    df = df[non_empty].copy()

    # Filtering step 2: Length checks
    char_lens = df["text"].str.len()
    valid_len = (char_lens >= min_char_len) & (char_lens <= max_char_len)
    filtering_stats["removed_too_short"] = int((char_lens < min_char_len).sum())
    filtering_stats["removed_too_long"] = int((char_lens > max_char_len).sum())
    df = df[valid_len].copy()

    # Filtering step 3: Exact duplicates removal
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    filtering_stats["removed_exact_duplicates"] = pre_dedup - len(df)

    final_count = len(df)
    logger.info(f"Cleaned dataset records count: {final_count}")

    # Compute text length statistics
    df["char_length"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    # Save processed dataset
    processed_path = os.path.join(processed_dir, "processed_ai_detection.parquet")
    df.to_parquet(processed_path, index=False)
    csv_processed_path = os.path.join(processed_dir, "processed_ai_detection.csv")
    df.to_csv(csv_processed_path, index=False)
    logger.info(f"Saved processed dataset to {processed_path} and {csv_processed_path}")

    # Data Quality Report
    class_distribution = df["label_name"].value_counts().to_dict()
    source_distribution = df["source_dataset"].value_counts().to_dict()
    generator_distribution = df["generator"].value_counts().to_dict()

    quality_report = {
        "module": "AI-Generated vs Human-Written Text Detection",
        "original_extracted_rows": original_count,
        "final_cleaned_rows": final_count,
        "filtering_statistics": filtering_stats,
        "class_distribution": class_distribution,
        "source_dataset_distribution": source_distribution,
        "generator_distribution": generator_distribution,
        "text_statistics": {
            "char_length_mean": float(df["char_length"].mean()),
            "char_length_std": float(df["char_length"].std()),
            "char_length_min": int(df["char_length"].min()),
            "char_length_max": int(df["char_length"].max()),
            "word_count_mean": float(df["word_count"].mean()),
            "word_count_std": float(df["word_count"].std()),
            "word_count_min": int(df["word_count"].min()),
            "word_count_max": int(df["word_count"].max())
        }
    }

    report_path = os.path.join(reports_dir, "ai_detection_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved data quality report to {report_path}")

    return df

if __name__ == "__main__":
    preprocess_ai_detection_data()
