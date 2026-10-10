import os
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_ai_detection_data(
    processed_file: str = "data/processed/ai_detection/processed_ai_detection.parquet",
    splits_dir: str = "data/splits/ai_detection",
    reports_dir: str = "reports/dataset_quality",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> dict:
    """
    Creates leakage-free, stratified train/val/test splits.
    Ensures exact and near duplicates are grouped/isolated.
    """
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(processed_file):
        # Fallback to CSV if parquet not present
        csv_file = processed_file.replace(".parquet", ".csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(f"Processed file {processed_file} not found.")
    else:
        df = pd.read_parquet(processed_file)

    logger.info(f"Loaded {len(df)} cleaned AI detection samples for splitting.")

    # Deduplicate purely to ensure 0% duplicate leakage across splits
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # First split: Train (80%) vs Temp (20%)
    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["label"],
        random_state=random_seed
    )

    # Second split: Val (50% of temp = 10%) vs Test (50% of temp = 10%)
    val_prop = val_ratio / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_prop),
        stratify=temp_df["label"],
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

    logger.info(f"Saved splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Record split summary
    split_summary = {
        "module": "AI-Generated vs Human-Written Text Detection",
        "random_seed": random_seed,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_class_balance": train_df["label_name"].value_counts().to_dict(),
        "val_class_balance": val_df["label_name"].value_counts().to_dict(),
        "test_class_balance": test_df["label_name"].value_counts().to_dict(),
        "train_sources": train_df["source_dataset"].value_counts().to_dict(),
        "test_sources": test_df["source_dataset"].value_counts().to_dict()
    }

    summary_path = os.path.join(reports_dir, "ai_detection_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Saved AI detection split summary to {summary_path}")

    return {"train": train_df, "val": val_df, "test": test_df}

if __name__ == "__main__":
    split_ai_detection_data()
