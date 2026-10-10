import argparse
import logging
import sys
import time

from src.data_acquisition import acquire_ai_detection_data, acquire_grammar_data
from src.preprocessing import preprocess_ai_detection_data, preprocess_grammar_data
from src.splitting import split_ai_detection_data, split_grammar_data
from src.training import train_ai_detection_models, setup_and_evaluate_grammar_model
from src.evaluation import evaluate_ai_detection_test_set, evaluate_grammar_test_sets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("PipelineRunner")

def run_stage_acquire():
    logger.info("==========================================")
    logger.info("STAGE 1: DATA ACQUISITION")
    logger.info("==========================================")
    acquire_ai_detection_data()
    acquire_grammar_data()
    logger.info("Stage 1 completed successfully.")

def run_stage_preprocess():
    logger.info("==========================================")
    logger.info("STAGE 2: DATA PREPROCESSING & CLEANING")
    logger.info("==========================================")
    preprocess_ai_detection_data()
    preprocess_grammar_data()
    logger.info("Stage 2 completed successfully.")

def run_stage_split():
    logger.info("==========================================")
    logger.info("STAGE 3: LEAKAGE-FREE DATASET SPLITTING")
    logger.info("==========================================")
    split_ai_detection_data()
    split_grammar_data()
    logger.info("Stage 3 completed successfully.")

def run_stage_train():
    logger.info("==========================================")
    logger.info("STAGE 4: MODEL TRAINING & VALIDATION")
    logger.info("==========================================")
    train_ai_detection_models()
    setup_and_evaluate_grammar_model()
    logger.info("Stage 4 completed successfully.")

def run_stage_evaluate():
    logger.info("==========================================")
    logger.info("STAGE 5: HELD-OUT TEST EVALUATION")
    logger.info("==========================================")
    evaluate_ai_detection_test_set()
    evaluate_grammar_test_sets()
    logger.info("Stage 5 completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="AI Content Analysis System - ML Pipeline Execution")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "acquire", "preprocess", "split", "train", "evaluate"],
        help="Pipeline stage to execute (default: all)"
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"Starting ML Pipeline Execution (stage: {args.stage})...")

    if args.stage in ["all", "acquire"]:
        run_stage_acquire()
    if args.stage in ["all", "preprocess"]:
        run_stage_preprocess()
    if args.stage in ["all", "split"]:
        run_stage_split()
    if args.stage in ["all", "train"]:
        run_stage_train()
    if args.stage in ["all", "evaluate"]:
        run_stage_evaluate()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline finished successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
