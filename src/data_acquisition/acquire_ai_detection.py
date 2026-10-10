import os
import json
import logging
from datetime import datetime
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def acquire_ai_detection_data(
    raw_dir: str = "data/raw/ai_detection",
    sample_size: int = 6000
) -> dict:
    """
    Acquires raw datasets for AI-Generated vs Human-Written text detection.
    Sources:
    1. 'aadityaubhat/GPT-wiki-intro' (GPT-3 vs Human Wikipedia introductions)
       - License: CC-BY-SA 4.0
       - Domain: General knowledge / encyclopedic
    2. 'Hello-SimpleAI/hc3' (ChatGPT vs Human answers across multiple domains)
       - License: CC-BY 4.0
       - Domain: Multi-domain QA (Reddit, Wiki, CS, Medicine, Finance)
    """
    os.makedirs(raw_dir, exist_ok=True)
    logger.info(f"Starting acquisition for Module A (AI vs Human Text Detection)...")

    # 1. Acquire GPT-wiki-intro
    logger.info("Downloading GPT-wiki-intro dataset from Hugging Face...")
    wiki_dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    # Extract human and generated pairs
    wiki_df = wiki_dataset.select(range(min(sample_size, len(wiki_dataset)))).to_pandas()
    wiki_raw_path = os.path.join(raw_dir, "raw_gpt_wiki_intro.jsonl")
    wiki_df.to_json(wiki_raw_path, orient="records", lines=True)
    logger.info(f"Saved {len(wiki_df)} raw GPT-wiki-intro records to {wiki_raw_path}")

    # 2. Acquire HC3 (Human ChatGPT Comparison Corpus)
    logger.info("Downloading HC3 dataset from Hugging Face...")
    hc3_dataset = load_dataset("Hello-SimpleAI/hc3", "default", split="train")
    hc3_df = hc3_dataset.select(range(min(sample_size, len(hc3_dataset)))).to_pandas()
    hc3_raw_path = os.path.join(raw_dir, "raw_hc3.jsonl")
    hc3_df.to_json(hc3_raw_path, orient="records", lines=True)
    logger.info(f"Saved {len(hc3_df)} raw HC3 records to {hc3_raw_path}")

    # Save acquisition metadata
    metadata = {
        "module": "AI-Generated vs Human-Written Text Detection",
        "acquisition_date": datetime.utcnow().isoformat(),
        "datasets": [
            {
                "name": "GPT-wiki-intro",
                "source_url": "https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro",
                "license": "CC-BY-SA 4.0",
                "original_samples": len(wiki_dataset),
                "downloaded_samples": len(wiki_df),
                "raw_file": wiki_raw_path,
                "text_generators": ["GPT-3 (Curie/Davinci)"],
                "human_sources": ["Wikipedia"],
                "known_limitations": "Primarily introductory expository style; high formal quality."
            },
            {
                "name": "HC3 (Human ChatGPT Comparison Corpus)",
                "source_url": "https://huggingface.co/datasets/Hello-SimpleAI/hc3",
                "license": "CC-BY 4.0",
                "original_samples": len(hc3_dataset),
                "downloaded_samples": len(hc3_df),
                "raw_file": hc3_raw_path,
                "text_generators": ["ChatGPT (GPT-3.5-turbo)"],
                "human_sources": ["Reddit ELI5", "Wikipedia QA", "OpenQA", "Finance", "Medicine"],
                "known_limitations": "Conversational & question-answering format; variable length answers."
            }
        ]
    }
    
    metadata_path = os.path.join(raw_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved dataset acquisition metadata to {metadata_path}")

    return metadata

if __name__ == "__main__":
    acquire_ai_detection_data()
