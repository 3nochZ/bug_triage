import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.models.base import BaseClassifier

RESULTS_DIR = Path("results")
TARGET_CLASSES = ["bug", "feature", "question", "docs"]


def evaluate_model(
    model: BaseClassifier,
    test_texts: List[str],
    test_labels: List[str],
    save_results: bool = True
) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    outputs = model.predict_batch(test_texts)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    preds = [o.label for o in outputs]
    avg_latency_ms = sum(o.latency_ms for o in outputs) / max(len(outputs), 1)
    avg_cost_usd = sum(o.cost_usd for o in outputs) / max(len(outputs), 1)

    macro_f1 = float(f1_score(test_labels, preds, labels=TARGET_CLASSES, average="macro", zero_division=0))
    report = classification_report(test_labels, preds, labels=TARGET_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_labels, preds, labels=TARGET_CLASSES).tolist()

    result = {
        "model_name": model.name,
        "sample_count": len(test_texts),
        "macro_f1": macro_f1,
        "accuracy": float(report.get("accuracy", 0.0)),
        "avg_latency_ms": avg_latency_ms,
        "avg_cost_usd": avg_cost_usd,
        "total_wall_clock_ms": total_time_ms,
        "confusion_matrix": {
            "labels": TARGET_CLASSES,
            "matrix": cm
        },
        "per_class": {
            cls_name: {
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "f1_score": report[cls_name]["f1-score"],
                "support": report[cls_name]["support"]
            }
            for cls_name in TARGET_CLASSES if cls_name in report
        }
    }

    if save_results:
        # persist metrics for comparison across milestones
        out_file = RESULTS_DIR / f"{model.name}_metrics.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def print_evaluation_summary(result: Dict[str, Any]):
    print(f"\n{'=' * 50}")
    print(f"Model: {result['model_name']}")
    print(f"{'=' * 50}")
    print(f"Macro-F1:            {result['macro_f1']:.4f}")
    print(f"Accuracy:            {result['accuracy']:.4f}")
    print(f"Avg Latency / issue: {result['avg_latency_ms']:.2f} ms")
    print(f"Avg Cost / issue:    ${result['avg_cost_usd']:.6f}")
    print("\nPer-Class Breakdown:")
    for cls_name, metrics in result["per_class"].items():
        print(f"  {cls_name:10s} -> P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f} (n={metrics['support']})")

    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    labels = result["confusion_matrix"]["labels"]
    matrix = result["confusion_matrix"]["matrix"]
    header = "          " + " ".join(f"{l:>10s}" for l in labels)
    print(header)
    for l, row in zip(labels, matrix):
        row_str = " ".join(f"{cnt:10d}" for cnt in row)
        print(f"{l:10s}{row_str}")
    print(f"{'=' * 50}\n")
