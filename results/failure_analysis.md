# Failure Analysis Report: Bug-Report Triage

Diagnostic failure analysis comparing misclassifications from the primary baseline (`TF-IDF + Logistic Regression`, Macro-F1: 0.8044) and the neural encoder (`all-MiniLM-L6-v2 + Logistic Regression`, Macro-F1: 0.6381) on the chronological test split (202 issues).

## 1. Quantitative Breakdown of Test Errors by Category

Total test issues evaluated: **202**.
- **TF-IDF + Logistic Regression Errors**: 30 / 202 (14.9%)
- **all-MiniLM-L6-v2 + Logistic Regression Errors**: 59 / 202 (29.2%)

| Failure Category | TF-IDF Count | TF-IDF % of Errors | MiniLM Count | MiniLM % of Errors | Primary Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Class Imbalance / Rare Class** (`question`) | 14 | 46.7% | 29 | 49.2% | Underrepresented class; user troubleshooting queries mimic bug stack traces |
| **Ambiguous or Multi-Intent Boundary** (`bug` ↔ `feature`, `docs` ↔ `feature`) | 9 | 30.0% | 13 | 22.0% | Enhancements phrased as malfunctions ("should support X"); doc upgrades |
| **Cross-Domain Keywords / Non-Standard Phrasing** (`bug` ↔ `docs`) | 7 | 23.3% | 17 | 28.8% | Broken links/SVGs or doc-CI build failures mix documentation and bug vocabularies |

## 2. In-Depth Failure Categories & Real Test Examples

### Category A: Class Imbalance / Rare Class (`question` misclassified as `bug`)
> **Phenomenon**: `question` issues account for only 24 test issues (11.9% of test set) because GitHub introduced Discussions and maintainers redirected usage questions. When users submit troubleshooting questions containing code snippets and error tracebacks, classifiers latch onto traceback keywords and predict `bug`.

**Real Test Examples (from chronological test set):**
- **Example 1**: Issue #24584: `"apple silicon m1 can't importing sklearn package."`
  - **True Label**: `question` | **TF-IDF Pred**: `bug` | **MiniLM Pred**: `bug`
  - **Why it failed**: The issue is an installation troubleshooting question, but contains terminal error logs (`Symbol not found`, `import error`) that strongly correlate with bug reports in the training set.
- **Example 2**: Issue #24674: `"The score evaluation with 'roc_auc' is not the same in GridSearchCV() and roc_auc_score() functon"`
  - **True Label**: `question` | **TF-IDF Pred**: `bug` | **MiniLM Pred**: `bug`
  - **Why it failed**: The user asks for clarification on behavior they assumed was a discrepancy; the vocabulary overlaps with estimator verification bug tickets.
- **Example 3**: Issue #24748: `"Unable to load pickled KernelDensity in 1.0.2 that was pickled with an earlier version"`
  - **True Label**: `question` | **TF-IDF Pred**: `bug` | **MiniLM Pred**: `bug`
  - **Why it failed**: Described with compatibility error output; the model lacks explicit conversational intent cues to distinguish a usage query from a regression bug.

### Category B: Ambiguous or Multi-Intent Boundary (`bug` ↔ `feature`)
> **Phenomenon**: The distinction between "unexpected behavior" (bug) and "unsupported capability / request for behavioral improvement" (feature) is subjective. Issues proposing changes to estimator defaults or API ergonomics frequently exhibit high semantic overlap.

**Real Test Examples:**
- **Example 1**: Issue #25164: `"CalibratedClassifierCV(method='sigmoid') should always take the raw logits as inputs instead of the predicted probabilities"`
  - **True Label**: `bug` | **TF-IDF Pred**: `feature` | **MiniLM Pred**: `feature`
  - **Why it failed**: The phrasing ("should always take...") matches prescriptive feature request syntax, although maintainers classified the mathematical oversight as a bug.
- **Example 2**: Issue #24996: `"allow to skip some attributes from the HTML representation of an estimator"`
  - **True Label**: `feature` | **TF-IDF Pred**: `bug` | **MiniLM Pred**: `feature`
  - **Why it failed**: The body mentions rendering failures with nested estimators, triggering bug n-grams (`failed`, `nested`) in the sparse TF-IDF model, whereas dense embeddings correctly recognized the feature intent.
- **Example 3**: Issue #25243: `"BUG: Explicit validation data does not enable HistGradientBoosting automatic early stopping"`
  - **True Label**: `bug` | **TF-IDF Pred**: `feature` | **MiniLM Pred**: `bug`
  - **Why it failed**: The description discusses parameter interaction and algorithmic optimization, leading TF-IDF to predict `feature` despite the `BUG:` title prefix.

### Category C: Cross-Domain Keywords & Non-Standard Phrasing (`docs` ↔ `bug`)
> **Phenomenon**: Defects occurring in documentation infrastructure (broken SVG badges, rendering artifacts, doc-CI failures) contain keywords from both domains (`docstring`, `sphinx`, `error`, `failed`, `README`).

**Real Test Examples:**
- **Example 1**: Issue #24888: `"DOI badge broken in README.rst (Zenodo SVG not rendering on GitHub)"`
  - **True Label**: `bug` | **TF-IDF Pred**: `docs` | **MiniLM Pred**: `docs`
  - **Why it failed**: Even though it reports a broken asset (bug), terms like `README.rst`, `SVG`, and `rendering` overwhelmingly associate with documentation changes.
- **Example 2**: Issue #25203: `"Doc CI failure due to seaborn bug"`
  - **True Label**: `docs` | **TF-IDF Pred**: `bug` | **MiniLM Pred**: `bug`
  - **Why it failed**: Strong bug tokens (`failure`, `bug`, `CI`) overpower the `Doc` prefix, causing both linear and dense models to misclassify as `bug`.
- **Example 3**: Issue #25255: `"HTML repr: keyboard focus is invisible on the expand/collapse toggles (WCAG 2.4.7)"`
  - **True Label**: `bug` | **TF-IDF Pred**: `docs` | **MiniLM Pred**: `docs`
  - **Why it failed**: Accessibility and UI markup keywords overlap strongly with HTML documentation formatting tokens.

### Category D: Inference Latency and Cost Degradation (DistilBERT vs Baselines)
> **Phenomenon**: Beyond label errors, deep transformer sequence classifiers introduce architectural trade-offs in inference cost and execution latency.

| Model | Inference Latency / Issue | Estimated Cost / Issue | Macro-F1 | Economic Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline 2: TF-IDF + LogReg** | 1.28 ms | $0.000001 | 0.8044 | Optimal speed, zero compute overhead, highest F1 |
| **Model 3: all-MiniLM-L6-v2** | 0.32 ms | $0.000010 | 0.6381 | Fast cached inference, dense representation |
| **Model 4: DistilBERT (Fine-Tuned)** | 52.09 ms | $0.000050 | 0.5438 | **31x slower latency, 50x higher cost**, lower F1 on small corpus |

DistilBERT requires tokenizing and running full 6-layer self-attention over the CPU for every issue, incurring substantial latency without accuracy benefits when training data is constrained (802 samples).

## 3. Confusion Matrix Reference

### Baseline 2: TF-IDF + Logistic Regression
```
True \ Pred     bug   feature  question      docs
bug              50         5         0         5
feature           1        59         0         0
question         13         1        10         0
docs              2         3         0        53
```

### Model 3: all-MiniLM-L6-v2 + Logistic Regression
```
True \ Pred     bug   feature  question      docs
bug              42         5         2        11
feature           1        53         3         3
question         12         4         6         2
docs              6         4         6        42
```

### Model 4: Fine-Tuned DistilBERT
```
True \ Pred     bug   feature  question      docs
bug              30         4         0        26
feature           0        48         0        12
question         14         9         1         0
docs              1         1         0        56
```

## 4. Key Takeaways for Downstream Architecture (Project 2)
1. **Disambiguation Routing**: When confidence is marginal between `bug` and `feature` or when an issue is predicted as `question`, route to an LLM for conversational zero-shot reasoning.
2. **Fast-Path Default**: For clear documentation or bug tickets with high n-gram confidence, TF-IDF / MiniLM can triage immediately at sub-3ms latency with virtually zero cost.