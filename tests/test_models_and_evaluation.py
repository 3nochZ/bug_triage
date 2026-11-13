import pytest
from src.dataset import TARGET_CLASSES
from src.models.base import BaseClassifier, PredictionOutput
from src.models.rule_based import RuleBasedClassifier
from src.models.tfidf import TfidfLogisticClassifier
from src.evaluate import evaluate_model


class MockClassifier(BaseClassifier):
    @property
    def name(self) -> str:
        return "mock_classifier"

    def fit(self, train_texts, train_labels) -> None:
        pass

    def predict_batch(self, texts):
        return [
            PredictionOutput(
                label="bug",
                latency_ms=1.25,
                cost_usd=0.000001,
                confidence=0.95
            )
            for _ in texts
        ]


def test_rule_based_classifier_contract():
    model = RuleBasedClassifier()
    out = model.predict_one("BUG: NullPointerException when running fit() with traceback")
    assert isinstance(out, PredictionOutput)
    assert out.label in TARGET_CLASSES
    assert out.label == "bug"
    assert out.latency_ms >= 0.0
    assert out.cost_usd == 0.0
    assert 0.0 <= out.confidence <= 1.0


def test_tfidf_classifier_contract():
    train_texts = [
        "AttributeError traceback bug in fit",
        "Add new feature to support sparse arrays",
        "How do I use this estimator parameter?",
        "Fix typo in README documentation guide"
    ]
    train_labels = ["bug", "feature", "question", "docs"]

    model = TfidfLogisticClassifier()
    model.fit(train_texts, train_labels)

    out = model.predict_one("There is a traceback bug in fit()")
    assert isinstance(out, PredictionOutput)
    assert out.label in TARGET_CLASSES
    assert out.latency_ms >= 0.0
    assert out.cost_usd >= 0.0
    assert 0.0 <= out.confidence <= 1.0


def test_evaluation_harness_contract():
    mock_model = MockClassifier()
    test_texts = ["sample issue 1", "sample issue 2", "sample issue 3", "sample issue 4"]
    test_labels = ["bug", "feature", "question", "docs"]

    # evaluate without overwriting persisted milestone results
    metrics = evaluate_model(mock_model, test_texts, test_labels, save_results=False)

    assert metrics["model_name"] == "mock_classifier"
    assert metrics["sample_count"] == 4
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["avg_latency_ms"] >= 0.0
    assert metrics["avg_cost_usd"] >= 0.0
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]["matrix"]) == len(TARGET_CLASSES)
    assert len(metrics["confusion_matrix"]["matrix"][0]) == len(TARGET_CLASSES)
