import os
import json
import logging
from datetime import datetime
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def acquire_grammar_data(
    raw_dir: str = "data/raw/grammar_correction",
    coedit_sample_size: int = 8000
) -> dict:
    """
    Acquires raw datasets for Grammar Error Detection and Correction.
    Sources:
    1. 'grammarly/coedit' (Grammarly GEC paired corpus)
       - License: CC-BY-NC-SA 4.0
       - Task: 'gec' (grammatical error correction)
    2. 'jhu-clsp/jfleg' (JFLEG Grammatical error correction benchmark test set)
       - License: Creative Commons Attribution 4.0 International (CC-BY 4.0)
       - Standard fluency & GEC evaluation benchmark
    """
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("Starting acquisition for Module B (Grammar Correction)...")

    # 1. Acquire Grammarly CoEdit GEC split
    logger.info("Downloading Grammarly CoEdit dataset from Hugging Face...")
    coedit_dataset = load_dataset("grammarly/coedit", split="train")
    
    # Filter for GEC task
    coedit_gec = [item for item in coedit_dataset if item.get("task") == "gec"]
    logger.info(f"Total GEC items in CoEdit train split: {len(coedit_gec)}")
    
    selected_coedit = coedit_gec[:min(coedit_sample_size, len(coedit_gec))]
    coedit_df = pd.DataFrame(selected_coedit)
    coedit_raw_path = os.path.join(raw_dir, "raw_coedit_gec.jsonl")
    coedit_df.to_json(coedit_raw_path, orient="records", lines=True)
    logger.info(f"Saved {len(coedit_df)} raw CoEdit GEC records to {coedit_raw_path}")

    # 2. Acquire JFLEG evaluation benchmark
    logger.info("Downloading JFLEG benchmark dataset from Hugging Face...")
    jfleg_test = load_dataset("jhu-clsp/jfleg", split="test")
    jfleg_df = jfleg_test.to_pandas()
    jfleg_raw_path = os.path.join(raw_dir, "raw_jfleg_test.jsonl")
    jfleg_df.to_json(jfleg_raw_path, orient="records", lines=True)
    logger.info(f"Saved {len(jfleg_df)} raw JFLEG benchmark records to {jfleg_raw_path}")

    # Save acquisition metadata
    metadata = {
        "module": "Grammar Error Detection and Correction",
        "acquisition_date": datetime.utcnow().isoformat(),
        "datasets": [
            {
                "name": "Grammarly CoEdit (GEC)",
                "source_url": "https://huggingface.co/datasets/grammarly/coedit",
                "license": "CC-BY-NC-SA 4.0",
                "original_samples": len(coedit_gec),
                "downloaded_samples": len(coedit_df),
                "raw_file": coedit_raw_path,
                "task": "Grammatical Error Correction (GEC)",
                "format": "src (incorrect with task prefix) -> tgt (corrected text)",
                "known_limitations": "Contains instruction prefixes; some sentence pairs may involve stylistic rephrasing."
            },
            {
                "name": "JFLEG Benchmark (JHU FLuency-Extended GUG)",
                "source_url": "https://huggingface.co/datasets/jhu-clsp/jfleg",
                "license": "CC-BY 4.0",
                "original_samples": len(jfleg_test),
                "downloaded_samples": len(jfleg_df),
                "raw_file": jfleg_raw_path,
                "task": "Fluency and Grammatical Error Correction Evaluation",
                "format": "sentence -> 4 reference corrections list",
                "known_limitations": "Standard benchmark test set; contains multi-reference corrections."
            }
        ]
    }
    
    metadata_path = os.path.join(raw_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved grammar dataset acquisition metadata to {metadata_path}")

    return metadata

if __name__ == "__main__":
    acquire_grammar_data()
