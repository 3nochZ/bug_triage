from src.dataset import load_splits
from src.evaluate import evaluate_model, print_evaluation_summary
from src.models.rule_based import RuleBasedClassifier
from src.models.tfidf import TfidfLogisticClassifier


def main():
    print("loading chronological train/test splits...")
    train_df, test_df = load_splits()

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"train set size: {len(train_texts)}, test set size: {len(test_texts)}")

    # evaluate baseline 1: rule-based classifier
    print("\nevaluating baseline 1 (rule-based keyword classifier)...")
    rule_model = RuleBasedClassifier()
    rule_model.fit(train_texts, train_labels)
    rule_results = evaluate_model(rule_model, test_texts, test_labels)
    print_evaluation_summary(rule_results)

    # evaluate baseline 2: tf-idf + logistic regression
    print("\nevaluating baseline 2 (tfidf + logistic regression)...")
    tfidf_model = TfidfLogisticClassifier()
    tfidf_model.fit(train_texts, train_labels)
    tfidf_results = evaluate_model(tfidf_model, test_texts, test_labels)
    print_evaluation_summary(tfidf_results)


if __name__ == "__main__":
    main()
