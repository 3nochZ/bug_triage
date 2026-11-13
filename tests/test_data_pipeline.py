import json
from pathlib import Path
import pandas as pd
import pytest

from src.dataset import map_labels, TARGET_CLASSES, TRAIN_PATH, TEST_PATH, RAW_DATA_PATH


def test_no_temporal_data_leakage():
    # load chronological splits to verify zero future data leakage
    assert TRAIN_PATH.exists(), f"missing train split at {TRAIN_PATH}"
    assert TEST_PATH.exists(), f"missing test split at {TEST_PATH}"

    train_df = pd.read_json(TRAIN_PATH, lines=True)
    test_df = pd.read_json(TEST_PATH, lines=True)

    # ensure no overlapping issue ids between splits
    train_ids = set(train_df["id"].tolist())
    test_ids = set(test_df["id"].tolist())
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"data leakage detected: {len(overlap)} overlapping ids"

    # chronological verification per target class
    for cls in TARGET_CLASSES:
        cls_train = train_df[train_df["label"] == cls]
        cls_test = test_df[test_df["label"] == cls]
        if not cls_train.empty and not cls_test.empty:
            max_train_date = cls_train["created_at"].max()
            min_test_date = cls_test["created_at"].min()
            # train timestamps must strictly predate or equal test timestamps
            assert max_train_date <= min_test_date, (
                f"temporal leakage in class {cls}: max train {max_train_date} > min test {min_test_date}"
            )


def test_pr_filtering():
    # pull requests must be excluded to prevent noisy code diffs
    if RAW_DATA_PATH.exists():
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            raw_issues = json.load(f)
        for issue in raw_issues:
            assert "pull_request" not in issue, f"issue {issue.get('number')} is a pull request"


def test_label_mapping_and_classes():
    # test label normalization across raw label variants
    assert map_labels({"labels": [{"name": "Bug: regression"}]}) == "bug"
    assert map_labels({"labels": [{"name": "Enhancement"}]}) == "feature"
    assert map_labels({"labels": [{"name": "Documentation"}]}) == "docs"
    assert map_labels({"labels": [{"name": "Question"}]}) == "question"
    assert map_labels({"labels": [{"name": "needs triage"}]}) == ""

    # test that processed datasets only contain the 4 valid classes
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    test_df = pd.read_json(TEST_PATH, lines=True)

    assert set(train_df["label"].unique()).issubset(set(TARGET_CLASSES))
    assert set(test_df["label"].unique()).issubset(set(TARGET_CLASSES))
