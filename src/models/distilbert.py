import time
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup
)

from src.models.base import BaseClassifier, PredictionOutput

SAVED_MODEL_DIR = Path("results/distilbert_saved")
TARGET_CLASSES = ["bug", "feature", "question", "docs"]


class IssueDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class DistilBertClassifier(BaseClassifier):
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(TARGET_CLASSES))
        self.label2id = {name: i for i, name in enumerate(TARGET_CLASSES)}
        self.id2label = {i: name for i, name in enumerate(TARGET_CLASSES)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def name(self) -> str:
        return "distilbert_finetuned"

    def fit(self, train_texts: List[str], train_labels: List[str], epochs: int = 2, batch_size: int = 16, lr: float = 3e-5) -> None:
        if SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "model.safetensors").exists():
            # load cached fine-tuned weights if previously trained
            self.model = DistilBertForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)
            self.model.to(self.device)
            return

        numeric_labels = [self.label2id[l] for l in train_labels]
        encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_length)
        dataset = IssueDataset(encodings, numeric_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"  epoch {epoch + 1}/{epochs} - average cross-entropy loss: {avg_loss:.4f}")

        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(SAVED_MODEL_DIR)
        self.tokenizer.save_pretrained(SAVED_MODEL_DIR)

    def predict_batch(self, texts: List[str]) -> List[PredictionOutput]:
        self.model.eval()
        outputs = []

        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)
        dataset = IssueDataset(encodings)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_confs = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_confs.extend(confs.cpu().tolist())

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        avg_latency_ms = total_time_ms / max(len(texts), 1)

        for pred_id, conf in zip(all_preds, all_confs):
            label = self.id2label[pred_id]
            # compute estimated inference cost for 66m-parameter distilbert forward pass
            cost_usd = 0.000050
            outputs.append(PredictionOutput(
                label=label,
                latency_ms=avg_latency_ms,
                cost_usd=cost_usd,
                confidence=float(conf)
            ))

        return outputs
