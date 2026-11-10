from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PredictionOutput:
    label: str
    latency_ms: float
    cost_usd: float
    confidence: Optional[float] = None


class BaseClassifier(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[PredictionOutput]:
        pass

    def predict_one(self, text: str) -> PredictionOutput:
        # standard single-item predictor contract for project 2 router
        return self.predict_batch([text])[0]
