import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_feature_extractor():
    """
    Builds a robust hybrid n-gram TF-IDF feature extractor:
    - Word n-grams (1, 2) to capture syntactic style & terminology
    - Character n-grams (3, 5) to capture sub-word morphology & generation artifacts
    """
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char",
            max_features=25000,
            sublinear_tf=True
        ))
    ])

def train_ai_detection_models(
    splits_dir: str = "data/splits/ai_detection",
    models_dir: str = "models/ai_detection",
    reports_dir: str = "reports",
    random_seed: int = 42
) -> dict:
    """
    Trains and compares multiple AI text detection models on train split,
    validates on validation split, selects best model, and saves artifacts & plots.
    """
    os.makedirs(models_dir, exist_ok=True)
    plots_dir = os.path.join(reports_dir, "plots")
    eval_reports_dir = os.path.join(reports_dir, "evaluation")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(eval_reports_dir, exist_ok=True)

    # Load splits
    train_file = os.path.join(splits_dir, "train.parquet")
    val_file = os.path.join(splits_dir, "val.parquet")

    if os.path.exists(train_file):
        train_df = pd.read_parquet(train_file)
        val_df = pd.read_parquet(val_file)
    else:
        train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))

    logger.info(f"Loaded train ({len(train_df)}) and val ({len(val_df)}) splits.")

    X_train, y_train = train_df["text"].values, train_df["label"].values
    X_val, y_val = val_df["text"].values, val_df["label"].values

    # Candidate models
    candidate_pipelines = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("features", build_feature_extractor()),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=random_seed))
        ]),
        "TF-IDF + Linear SVM (SGD)": Pipeline([
            ("features", build_feature_extractor()),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=random_seed))
        ]),
        "TF-IDF + Multinomial Naive Bayes": Pipeline([
            ("features", build_feature_extractor()),
            ("clf", MultinomialNB(alpha=0.1))
        ])
    }

    validation_results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipeline in candidate_pipelines.items():
        logger.info(f"Training model: {name}...")
        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]

        acc = float(accuracy_score(y_val, y_val_pred))
        prec = float(precision_score(y_val, y_val_pred))
        rec = float(recall_score(y_val, y_val_pred))
        f1 = float(f1_score(y_val, y_val_pred))
        roc_auc = float(roc_auc_score(y_val, y_val_prob))

        logger.info(f"Validation -> {name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        validation_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    logger.info(f"Best model selected: {best_name} with Validation F1 = {best_f1:.4f}")

    # Save best model pipeline
    best_model_path = os.path.join(models_dir, "best_ai_detector.joblib")
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"Saved best model pipeline to {best_model_path}")

    # Save metadata
    model_metadata = {
        "best_model_name": best_name,
        "selected_metric": "f1_score",
        "validation_f1": best_f1,
        "all_candidates_validation": validation_results,
        "classes": ["human", "ai"],
        "label_mapping": {0: "human", 1: "ai"},
        "model_path": best_model_path
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Generate validation ROC Curve & Confusion Matrix Plots
    y_val_pred = best_pipeline.predict(X_val)
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    
    # 1. Confusion matrix plot
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.title(f"Validation Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "ai_detection_val_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # 2. ROC curve plot
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {model_metadata['all_candidates_validation'][best_name]['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Validation ROC Curve - {best_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(plots_dir, "ai_detection_val_roc_curve.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    logger.info(f"Plots saved to {cm_plot_path} and {roc_plot_path}")

    return {
        "best_model_name": best_name,
        "validation_results": validation_results,
        "best_model_path": best_model_path
    }

if __name__ == "__main__":
    train_ai_detection_models()
