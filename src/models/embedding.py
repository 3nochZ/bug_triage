import time
from pathlib import Path
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseClassifier, PredictionOutput

CACHE_DIR = Path("data/cache")


class EmbeddingClassifier(BaseClassifier):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._encoder: Optional[SentenceTransformer] = None
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.classes_ = []

    @property
    def name(self) -> str:
        return "encoder_minilm_l6_v2"

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _get_embeddings(self, texts: List[str], cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = CACHE_DIR / f"{cache_key}.npy"
            if cache_file.exists():
                # load precomputed embeddings to make iteration instant
                return np.load(cache_file)

        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if cache_key:
            np.save(CACHE_DIR / f"{cache_key}.npy", embeddings)

        return embeddings

    def fit(self, train_texts: List[str], train_labels: List[str], cache_key: str = "train_minilm") -> None:
        x_train = self._get_embeddings(train_texts, cache_key=cache_key)
        self.model.fit(x_train, train_labels)
        self.classes_ = list(self.model.classes_)

    def predict_batch(self, texts: List[str], cache_key: Optional[str] = "test_minilm") -> List[PredictionOutput]:
        t0 = time.perf_counter()
        embeddings = self._get_embeddings(texts, cache_key=cache_key)
        total_embed_time_ms = (time.perf_counter() - t0) * 1000.0

        per_item_embed_ms = total_embed_time_ms / max(len(texts), 1)

        outputs = []
        for emb in embeddings:
            t1 = time.perf_counter()
            probs = self.model.predict_proba([emb])[0]
            best_idx = probs.argmax()
            label = self.classes_[best_idx]
            conf = float(probs[best_idx])
            clf_time_ms = (time.perf_counter() - t1) * 1000.0

            # compute estimated cost for small 22m-parameter encoder inference
            cost_usd = 0.000010
            outputs.append(PredictionOutput(
                label=label,
                latency_ms=per_item_embed_ms + clf_time_ms,
                cost_usd=cost_usd,
                confidence=conf
            ))
        return outputs
