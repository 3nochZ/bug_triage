# Bug-Report Triage System: Final Benchmark Report

Benchmark evaluating 4 approaches on chronological test split (202 issues) from `scikit-learn/scikit-learn`.

## 1. Overall Performance & Economics Summary

| Approach | Macro-F1 | Accuracy | Latency (ms/issue) | Est. Cost ($/issue) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Deterministic Keyword Rules** | 0.5554 | 0.6683 | 0.46 ms | $0.000000 |
| **Baseline 2: TF-IDF + Logistic Regression** | 0.8044 | 0.8515 | 1.28 ms | $0.000001 |
| **Model 3: all-MiniLM-L6-v2 + Logistic Regression** | 0.6381 | 0.7079 | 0.32 ms | $0.000010 |
| **Model 4: Fine-Tuned DistilBERT (distilbert-base-uncased)** | 0.5438 | 0.6683 | 52.09 ms | $0.000050 |

## 2. Per-Class F1 Breakdown

| Approach | Bug F1 | Feature F1 | Question F1 | Docs F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Deterministic Keyword Rules** | 0.692 | 0.694 | 0.067 | 0.769 |
| **Baseline 2: TF-IDF + Logistic Regression** | 0.794 | 0.922 | 0.588 | 0.914 |
| **Model 3: all-MiniLM-L6-v2 + Logistic Regression** | 0.694 | 0.841 | 0.293 | 0.724 |
| **Model 4: Fine-Tuned DistilBERT (distilbert-base-uncased)** | 0.571 | 0.787 | 0.080 | 0.737 |

## 3. Confusion Matrices

### Baseline 1: Deterministic Keyword Rules
| True \ Pred | bug | feature | question | docs |
| :--- | :---: | :---: | :---: | :---: |
| **bug** | 55 | 1 | 0 | 4 |
| **feature** | 15 | 34 | 3 | 8 |
| **question** | 20 | 1 | 1 | 2 |
| **docs** | 9 | 2 | 2 | 45 |

### Baseline 2: TF-IDF + Logistic Regression
| True \ Pred | bug | feature | question | docs |
| :--- | :---: | :---: | :---: | :---: |
| **bug** | 50 | 5 | 0 | 5 |
| **feature** | 1 | 59 | 0 | 0 |
| **question** | 13 | 1 | 10 | 0 |
| **docs** | 2 | 3 | 0 | 53 |

### Model 3: all-MiniLM-L6-v2 + Logistic Regression
| True \ Pred | bug | feature | question | docs |
| :--- | :---: | :---: | :---: | :---: |
| **bug** | 42 | 5 | 2 | 11 |
| **feature** | 1 | 53 | 3 | 3 |
| **question** | 12 | 4 | 6 | 2 |
| **docs** | 6 | 4 | 6 | 42 |

### Model 4: Fine-Tuned DistilBERT (distilbert-base-uncased)
| True \ Pred | bug | feature | question | docs |
| :--- | :---: | :---: | :---: | :---: |
| **bug** | 30 | 4 | 0 | 26 |
| **feature** | 0 | 48 | 0 | 12 |
| **question** | 14 | 9 | 1 | 0 |
| **docs** | 1 | 1 | 0 | 56 |

## 4. Key Engineering Insights & Project 2 Readiness

- **Speed & Cost Pareto Frontier**: Baseline 2 (TF-IDF + LogReg) achieves exceptional Macro-F1 (0.804) with sub-3ms latency and negligible compute cost.
- **Encoder Strengths**: Sentence-Transformers `all-MiniLM-L6-v2` provides dense semantic representations (384-d) that capture phrasing semantics without token vocabulary mismatch.
- **Project 2 Router Plug-in**: Every model adheres to the `BaseClassifier` contract with `predict_one(text) -> PredictionOutput(label, latency_ms, cost_usd, confidence)`. The downstream Project 2 router can directly evaluate `reward = (pred == ground_truth) - cost_usd` to decide between fast local encoder inference vs calling an LLM.
