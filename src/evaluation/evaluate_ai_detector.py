import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

def evaluate_ai_detection_test_set(
    model_path: str = "models/ai_detection/best_ai_detector.joblib",
    splits_dir: str = "data/splits/ai_detection",
    reports_dir: str = "reports"
) -> dict:
    """
    Evaluates the saved best AI detector on the held-out test split,
    computes global & domain/generator-separated metrics, and saves plots.
    """
    eval_dir = os.path.join(reports_dir, "evaluation")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train model first.")

    logger.info(f"Loading trained model from {model_path}...")
    pipeline = joblib.load(model_path)

    test_file = os.path.join(splits_dir, "test.parquet")
    if os.path.exists(test_file):
        test_df = pd.read_parquet(test_file)
    else:
        test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    logger.info(f"Loaded held-out test set with {len(test_df)} samples.")

    X_test = test_df["text"].values
    y_test = test_df["label"].values

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Global Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cls_report = classification_report(y_test, y_pred, target_names=["Human", "AI"], output_dict=True)

    logger.info("=== HELD-OUT TEST EVALUATION RESULTS (Module A) ===")
    logger.info(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")

    # Subgroup / Source-Separated Evaluation
    subgroup_metrics = {}
    for src in test_df["source_dataset"].unique():
        sub_df = test_df[test_df["source_dataset"] == src]
        sub_X = sub_df["text"].values
        sub_y = sub_df["label"].values
        sub_pred = pipeline.predict(sub_X)
        sub_prob = pipeline.predict_proba(sub_X)[:, 1]

        subgroup_metrics[src] = {
            "samples": len(sub_df),
            "accuracy": float(accuracy_score(sub_y, sub_pred)),
            "precision": float(precision_score(sub_y, sub_pred)),
            "recall": float(recall_score(sub_y, sub_pred)),
            "f1_score": float(f1_score(sub_y, sub_pred)),
            "roc_auc": float(roc_auc_score(sub_y, sub_prob))
        }
        logger.info(f"Subgroup '{src}' ({len(sub_df)} samples) -> Accuracy: {subgroup_metrics[src]['accuracy']:.4f}, F1: {subgroup_metrics[src]['f1_score']:.4f}")

    # Save Full Evaluation Report
    eval_report = {
        "module": "Module A: AI-Generated vs Human-Written Text Detection",
        "model_file": model_path,
        "test_samples": len(test_df),
        "overall_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        },
        "confusion_matrix": {
            "true_negatives_human": int(tn),
            "false_positives_ai": int(fp),
            "false_negatives_human": int(fn),
            "true_positives_ai": int(tp)
        },
        "per_class_classification_report": cls_report,
        "subgroup_source_evaluations": subgroup_metrics
    }

    report_path = os.path.join(eval_dir, "ai_detection_test_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Saved test evaluation report to {report_path}")

    # 1. Confusion Matrix Heatmap Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Human", "Predicted AI"], yticklabels=["Actual Human", "Actual AI"])
    plt.title("Held-out Test Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "ai_detection_test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Test ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Held-out Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "ai_detection_test_roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    logger.info(f"Test plots saved to {cm_path} and {roc_path}")
    return eval_report

if __name__ == "__main__":
    evaluate_ai_detection_test_set()
