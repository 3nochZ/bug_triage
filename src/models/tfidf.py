import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseClassifier, PredictionOutput


class TfidfLogisticClassifier(BaseClassifier):
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.classes_ = []

    @property
    def name(self) -> str:
        return "baseline_2_tfidf"

    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        x_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(x_train, train_labels)
        self.classes_ = list(self.model.classes_)

    def predict_batch(self, texts: List[str]) -> List[PredictionOutput]:
        outputs = []
        for text in texts:
            t0 = time.perf_counter()
            x = self.vectorizer.transform([text])
            probs = self.model.predict_proba(x)[0]
            best_idx = probs.argmax()
            label = self.classes_[best_idx]
            conf = float(probs[best_idx])
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # compute cost for cpu matrix multiplication estimate
            cost_usd = 0.000001
            outputs.append(PredictionOutput(
                label=label,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                confidence=conf
            ))
        return outputs
