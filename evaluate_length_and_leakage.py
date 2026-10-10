import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

def analyze_leakage_and_length():
    splits_dir = "data/splits/ai_detection"
    model_path = "models/ai_detection/best_ai_detector.joblib"
    
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))
    
    print("=" * 60)
    print("1. DATASET OVERLAP & LEAKAGE AUDIT")
    print("=" * 60)
    
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    
    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))
    
    print(f"Total samples -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Exact Text Overlap (Train AND Val) : {train_val_overlap} samples ({(train_val_overlap/len(val_df))*100:.2f}%)")
    print(f"Exact Text Overlap (Train AND Test): {train_test_overlap} samples ({(train_test_overlap/len(test_df))*100:.2f}%)")
    print(f"Exact Text Overlap (Val AND Test)  : {val_test_overlap} samples ({(val_test_overlap/len(test_df))*100:.2f}%)")

    # Near-duplicate analysis: 50-character prefix collision check
    train_prefixes = set(t[:50].lower() for t in train_texts)
    val_prefixes = set(t[:50].lower() for t in val_texts)
    test_prefixes = set(t[:50].lower() for t in test_texts)

    prefix_train_val = len(train_prefixes.intersection(val_prefixes))
    prefix_train_test = len(train_prefixes.intersection(test_prefixes))

    print(f"Near-Duplicate 50-char Prefix Collisions (Train AND Val) : {prefix_train_val} ({(prefix_train_val/len(val_df))*100:.2f}%)")
    print(f"Near-Duplicate 50-char Prefix Collisions (Train AND Test): {prefix_train_test} ({(prefix_train_test/len(test_df))*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("2. PERFORMANCE BY TEXT LENGTH (WORD COUNT RANGES)")
    print("=" * 60)
    
    # Load model
    pipeline = joblib.load(model_path)
    
    test_df["word_count"] = test_df["text"].apply(lambda t: len(t.split()))
    test_df["char_len"] = test_df["text"].str.len()
    
    # Predict on entire test set
    X_test = test_df["text"].values
    y_test = test_df["label"].values
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    test_df["pred"] = y_pred
    test_df["prob_ai"] = y_prob
    test_df["prob_human"] = 1.0 - y_prob
    test_df["confidence"] = np.maximum(y_prob, 1.0 - y_prob)
    
    # Define length buckets
    buckets = [
        ("< 50 words", test_df[test_df["word_count"] < 50]),
        ("50 - 100 words", test_df[(test_df["word_count"] >= 50) & (test_df["word_count"] <= 100)]),
        ("> 100 words", test_df[test_df["word_count"] > 100]),
        # Granular sub-ranges
        ("< 30 words (very short)", test_df[test_df["word_count"] < 30]),
        ("30 - 50 words (short)", test_df[(test_df["word_count"] >= 30) & (test_df["word_count"] < 50)]),
    ]
    
    length_evaluation = {}
    
    for bucket_name, sub_df in buckets:
        if len(sub_df) == 0:
            continue
        sub_y = sub_df["label"].values
        sub_pred = sub_df["pred"].values
        sub_prob = sub_df["prob_ai"].values
        
        acc = float(accuracy_score(sub_y, sub_pred))
        prec = float(precision_score(sub_y, sub_pred, zero_division=0))
        rec = float(recall_score(sub_y, sub_pred, zero_division=0))
        f1 = float(f1_score(sub_y, sub_pred, zero_division=0))
        
        try:
            roc_auc = float(roc_auc_score(sub_y, sub_prob))
        except Exception:
            roc_auc = 0.0
            
        brier = float(brier_score_loss(sub_y, sub_prob))
        cm = confusion_matrix(sub_y, sub_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        avg_conf = float(sub_df["confidence"].mean())
        
        print(f"\n--- Bucket: {bucket_name} ({len(sub_df)} samples) ---")
        print(f"Class Balance : Human={int((sub_y==0).sum())}, AI={int((sub_y==1).sum())}")
        print(f"Accuracy      : {acc*100:.2f}%")
        print(f"Precision     : {prec:.4f}")
        print(f"Recall        : {rec:.4f}")
        print(f"F1-Score      : {f1:.4f}")
        print(f"ROC-AUC       : {roc_auc:.4f}")
        print(f"Brier Score   : {brier:.4f} (lower is better, 0 = perfect calibration)")
        print(f"Avg Confidence: {avg_conf*100:.2f}%")
        print(f"Confusion Matrix:")
        print(f"  [Actual Human] TN={tn:4d} | FP={fp:4d}")
        print(f"  [Actual AI   ] FN={fn:4d} | TP={tp:4d}")
        
        length_evaluation[bucket_name] = {
            "samples": len(sub_df),
            "human_count": int((sub_y == 0).sum()),
            "ai_count": int((sub_y == 1).sum()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "avg_confidence": avg_conf,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }
        
    print("\n" + "=" * 60)
    print("3. EXPLAINING 11-WORD SHORT SENTENCE BEHAVIOR")
    print("=" * 60)
    
    sample_11_words = "The students gathered in the quiet library to study for their exams."
    res_11 = pipeline.predict_proba([sample_11_words])[0]
    print(f"Sample 11-word text: '{sample_11_words}'")
    print(f"Human Prob: {res_11[0]*100:.2f}%, AI Prob: {res_11[1]*100:.2f}%")
    
    # Save report
    os.makedirs("reports/evaluation", exist_ok=True)
    with open("reports/evaluation/ai_detection_length_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(length_evaluation, f, indent=2)
    print("\nSaved length breakdown to reports/evaluation/ai_detection_length_breakdown.json")

if __name__ == "__main__":
    analyze_leakage_and_length()
