import json
from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/github_issues_raw.json")
PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"

TARGET_CLASSES = ["bug", "feature", "question", "docs"]


def map_labels(issue: dict) -> str:
    assigned = issue.get("_target_class")
    if assigned in TARGET_CLASSES:
        return assigned

    raw_labels = [l.get("name", "").lower() for l in issue.get("labels", [])]
    for name in raw_labels:
        if "bug" in name:
            return "bug"
        if "feature" in name or "enhancement" in name:
            return "feature"
        if "doc" in name:
            return "docs"
        if "question" in name:
            return "question"
    return ""


def clean_text(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    return f"{title}\n\n{body}".strip()


def prepare_dataset(split_ratio: float = 0.8, force: bool = False):
    if TRAIN_PATH.exists() and TEST_PATH.exists() and not force:
        train_df = pd.read_json(TRAIN_PATH, lines=True)
        test_df = pd.read_json(TEST_PATH, lines=True)
        return train_df, test_df

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}. Run src/fetcher.py first.")

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    records = []
    for item in raw_items:
        # exclude pull requests if any contaminated the raw dump
        if "pull_request" in item:
            continue

        label = map_labels(item)
        if label not in TARGET_CLASSES:
            continue

        text = clean_text(item.get("title", ""), item.get("body", ""))
        if not text:
            continue

        records.append({
            "id": item["id"],
            "number": item.get("number"),
            "created_at": item["created_at"],
            "title": item.get("title", ""),
            "text": text,
            "label": label,
        })

    df = pd.DataFrame(records)
    df["created_at_dt"] = pd.to_datetime(df["created_at"])

    # split chronologically per class to prevent temporal data leakage while preserving class representation
    train_records = []
    test_records = []

    for label_class in TARGET_CLASSES:
        class_subset = df[df["label"] == label_class].sort_values("created_at_dt").reset_index(drop=True)
        split_idx = int(len(class_subset) * split_ratio)
        train_records.append(class_subset.iloc[:split_idx])
        test_records.append(class_subset.iloc[split_idx:])

    train_df = pd.concat(train_records).sort_values("created_at_dt").reset_index(drop=True)
    test_df = pd.concat(test_records).sort_values("created_at_dt").reset_index(drop=True)

    train_df = train_df.drop(columns=["created_at_dt"])
    test_df = test_df.drop(columns=["created_at_dt"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_json(TRAIN_PATH, orient="records", lines=True)
    test_df.to_json(TEST_PATH, orient="records", lines=True)

    print(f"Total dataset: {len(df)} issues")
    print(f"Train split: {len(train_df)} ({train_df['created_at'].min()} to {train_df['created_at'].max()})")
    print(f"Test split:  {len(test_df)} ({test_df['created_at'].min()} to {test_df['created_at'].max()})")
    print("\nClass distribution in Train:")
    print(train_df["label"].value_counts().to_string())
    print("\nClass distribution in Test:")
    print(test_df["label"].value_counts().to_string())

    return train_df, test_df


def load_splits():
    if not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        return prepare_dataset()
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    test_df = pd.read_json(TEST_PATH, lines=True)
    return train_df, test_df


if __name__ == "__main__":
    prepare_dataset(force=True)
