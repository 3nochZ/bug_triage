import re
import time
from typing import List
from src.models.base import BaseClassifier, PredictionOutput


class RuleBasedClassifier(BaseClassifier):
    def __init__(self):
        # compile keyword patterns for quick matching
        self.patterns = {
            "bug": re.compile(r"\b(bug|error|exception|fail|failed|crash|traceback|broken|nan|segfault|unexpected|regression)\b", re.IGNORECASE),
            "docs": re.compile(r"\b(documentation|docstring|typo|user guide|tutorial|readme|examples?|docs/|guide|manual)\b", re.IGNORECASE),
            "feature": re.compile(r"\b(feature|enhancement|support for|add|implement|proposal|allow|expose|new algorithm)\b", re.IGNORECASE),
            "question": re.compile(r"\b(how to|how do i|is there a way|help|question|wondering|usage|best practice|clarification)\b", re.IGNORECASE),
        }

    @property
    def name(self) -> str:
        return "baseline_1_rules"

    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        # deterministic rules require no parameter fitting
        pass

    def _classify_single(self, text: str) -> str:
        scores = {k: len(pat.findall(text)) for k, pat in self.patterns.items()}
        max_score = max(scores.values())

        if max_score == 0:
            # default to most prevalent class when no keyword fires
            return "bug"

        # tie-break order: bug > docs > feature > question
        priority = ["bug", "docs", "feature", "question"]
        for p in priority:
            if scores[p] == max_score:
                return p
        return "bug"

    def predict_batch(self, texts: List[str]) -> List[PredictionOutput]:
        outputs = []
        for text in texts:
            t0 = time.perf_counter()
            label = self._classify_single(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            # rule evaluation uses negligible cpu and zero api tokens
            outputs.append(PredictionOutput(
                label=label,
                latency_ms=latency_ms,
                cost_usd=0.0,
                confidence=1.0
            ))
        return outputs
