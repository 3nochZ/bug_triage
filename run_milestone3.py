import json
from pathlib import Path
from src.dataset import load_splits
from src.evaluate import evaluate_model, print_evaluation_summary
from src.models.embedding import EmbeddingClassifier

RESULTS_DIR = Path("results")


def main():
    print("loading chronological train/test splits...")
    train_df, test_df = load_splits()

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"train set size: {len(train_texts)}, test set size: {len(test_texts)}")

    print("\nevaluating model 3 (sentence-transformers/all-MiniLM-L6-v2 embeddings + LogisticRegression)...")
    clf = EmbeddingClassifier()
    clf.fit(train_texts, train_labels, cache_key="train_minilm")
    results = evaluate_model(clf, test_texts, test_labels)
    print_evaluation_summary(results)

    # compare against baselines
    print("\n" + "=" * 65)
    print("INCREMENTAL COMPARISON: BASELINES VS ENCODER (MiniLM)")
    print("=" * 65)
    header = f"{'Model':<28} | {'Macro-F1':<10} | {'Latency/issue':<14} | {'Cost/issue':<12}"
    print(header)
    print("-" * 65)

    all_models = [
        ("baseline_1_rules", "Baseline 1 (Keyword Rules)"),
        ("baseline_2_tfidf", "Baseline 2 (TF-IDF + LogReg)"),
        ("encoder_minilm_l6_v2", "Model 3 (MiniLM + LogReg)"),
    ]

    for key, display_name in all_models:
        metric_file = RESULTS_DIR / f"{key}_metrics.json"
        if metric_file.exists():
            with open(metric_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"{display_name:<28} | {d['macro_f1']:<10.4f} | {d['avg_latency_ms']:<11.2f} ms | ${d['avg_cost_usd']:<11.6f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
