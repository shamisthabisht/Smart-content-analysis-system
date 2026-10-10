import os
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_grammar_data(
    processed_coedit_file: str = "data/processed/grammar_correction/processed_coedit_gec.parquet",
    processed_jfleg_file: str = "data/processed/grammar_correction/processed_jfleg_test.parquet",
    splits_dir: str = "data/splits/grammar_correction",
    reports_dir: str = "reports/dataset_quality",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> dict:
    """
    Creates leakage-free train/val/test splits for grammar error correction pairs,
    and isolates the JFLEG benchmark set for independent zero-shot evaluation.
    """
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(processed_coedit_file):
        csv_file = processed_coedit_file.replace(".parquet", ".csv")
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_parquet(processed_coedit_file)

    logger.info(f"Loaded {len(df)} processed CoEdit GEC pairs for splitting.")

    # Deduplicate source sentences to prevent split contamination
    df = df.drop_duplicates(subset=["source_text"]).reset_index(drop=True)

    # First split: Train (80%) vs Temp (20%)
    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_seed
    )

    # Second split: Val (50% of temp = 10%) vs Test (50% of temp = 10%)
    val_prop = val_ratio / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_prop),
        random_state=random_seed
    )

    # Save split files
    train_path = os.path.join(splits_dir, "train.parquet")
    val_path = os.path.join(splits_dir, "val.parquet")
    test_path = os.path.join(splits_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # Also link the JFLEG benchmark test set
    if os.path.exists(processed_jfleg_file):
        jfleg_df = pd.read_parquet(processed_jfleg_file)
    else:
        jfleg_csv = processed_jfleg_file.replace(".parquet", ".csv")
        jfleg_df = pd.read_csv(jfleg_csv)

    jfleg_split_path = os.path.join(splits_dir, "jfleg_benchmark_test.parquet")
    jfleg_df.to_parquet(jfleg_split_path, index=False)
    jfleg_df.to_csv(os.path.join(splits_dir, "jfleg_benchmark_test.csv"), index=False)

    logger.info(f"Saved Grammar splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, JFLEG Benchmark: {len(jfleg_df)}")

    split_summary = {
        "module": "Grammar Error Detection and Correction",
        "random_seed": random_seed,
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "test_pairs": len(test_df),
        "jfleg_benchmark_samples": len(jfleg_df),
        "split_strategy": "Non-overlapping source sentences split with fixed random seed."
    }

    summary_path = os.path.join(reports_dir, "grammar_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Saved Grammar split summary to {summary_path}")

    return {"train": train_df, "val": val_df, "test": test_df, "jfleg": jfleg_df}

if __name__ == "__main__":
    split_grammar_data()
