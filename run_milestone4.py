import json
from pathlib import Path
from src.dataset import load_splits
from src.evaluate import evaluate_model, print_evaluation_summary
from src.models.distilbert import DistilBertClassifier

RESULTS_DIR = Path("results")


def generate_final_report():
    model_keys = [
        ("baseline_1_rules", "Baseline 1: Deterministic Keyword Rules"),
        ("baseline_2_tfidf", "Baseline 2: TF-IDF + Logistic Regression"),
        ("encoder_minilm_l6_v2", "Model 3: all-MiniLM-L6-v2 + Logistic Regression"),
        ("distilbert_finetuned", "Model 4: Fine-Tuned DistilBERT (distilbert-base-uncased)"),
    ]

    all_data = {}
    for key, display_name in model_keys:
        metric_file = RESULTS_DIR / f"{key}_metrics.json"
        if metric_file.exists():
            with open(metric_file, "r", encoding="utf-8") as f:
                all_data[key] = (display_name, json.load(f))

    # export aggregated metrics json
    with open(RESULTS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: d[1] for k, d in all_data.items()}, f, indent=2)

    # generate markdown summary report
    md = []
    md.append("# Bug-Report Triage System: Final Benchmark Report\n")
    md.append("Benchmark evaluating 4 approaches on chronological test split (202 issues) from `scikit-learn/scikit-learn`.\n")

    md.append("## 1. Overall Performance & Economics Summary\n")
    md.append("| Approach | Macro-F1 | Accuracy | Latency (ms/issue) | Est. Cost ($/issue) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")

    for key, (display_name, d) in all_data.items():
        md.append(f"| **{display_name}** | {d['macro_f1']:.4f} | {d['accuracy']:.4f} | {d['avg_latency_ms']:.2f} ms | ${d['avg_cost_usd']:.6f} |")

    md.append("\n## 2. Per-Class F1 Breakdown\n")
    md.append("| Approach | Bug F1 | Feature F1 | Question F1 | Docs F1 |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")

    for key, (display_name, d) in all_data.items():
        pc = d["per_class"]
        bug_f1 = pc.get("bug", {}).get("f1_score", 0.0)
        feat_f1 = pc.get("feature", {}).get("f1_score", 0.0)
        q_f1 = pc.get("question", {}).get("f1_score", 0.0)
        docs_f1 = pc.get("docs", {}).get("f1_score", 0.0)
        md.append(f"| **{display_name}** | {bug_f1:.3f} | {feat_f1:.3f} | {q_f1:.3f} | {docs_f1:.3f} |")

    md.append("\n## 3. Confusion Matrices\n")
    for key, (display_name, d) in all_data.items():
        cm = d["confusion_matrix"]
        labels = cm["labels"]
        matrix = cm["matrix"]
        md.append(f"### {display_name}")
        md.append("| True \\ Pred | " + " | ".join(labels) + " |")
        md.append("| :--- | " + " | ".join([":---:"] * len(labels)) + " |")
        for l, row in zip(labels, matrix):
            md.append(f"| **{l}** | " + " | ".join(str(cnt) for cnt in row) + " |")
        md.append("")

    md.append("## 4. Key Engineering Insights & Project 2 Readiness\n")
    md.append("- **Speed & Cost Pareto Frontier**: Baseline 2 (TF-IDF + LogReg) achieves exceptional Macro-F1 (0.804) with sub-3ms latency and negligible compute cost.")
    md.append("- **Encoder Strengths**: Sentence-Transformers `all-MiniLM-L6-v2` provides dense semantic representations (384-d) that capture phrasing semantics without token vocabulary mismatch.")
    md.append("- **Project 2 Router Plug-in**: Every model adheres to the `BaseClassifier` contract with `predict_one(text) -> PredictionOutput(label, latency_ms, cost_usd, confidence)`. The downstream Project 2 router can directly evaluate `reward = (pred == ground_truth) - cost_usd` to decide between fast local encoder inference vs calling an LLM.\n")

    report_content = "\n".join(md)
    report_path = RESULTS_DIR / "final_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nFinal report successfully written to {report_path}")
    print("\n" + report_content)


def main():
    print("loading chronological train/test splits...")
    train_df, test_df = load_splits()

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"train set size: {len(train_texts)}, test set size: {len(test_texts)}")

    print("\nfine-tuning distilbert classifier (2 epochs on cpu)...")
    clf = DistilBertClassifier()
    clf.fit(train_texts, train_labels, epochs=2, batch_size=16, lr=3e-5)

    print("\nevaluating fine-tuned distilbert...")
    results = evaluate_model(clf, test_texts, test_labels)
    print_evaluation_summary(results)

    # generate full comparative benchmark report
    generate_final_report()


if __name__ == "__main__":
    main()
