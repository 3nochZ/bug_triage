# Bug-Report Triage System (Scenario 1: Encoder)

An automated GitHub issue classification system for open-source maintainers, categorizing issue `title + body` into four operational labels: `bug`, `feature`, `question`, or `docs`.

Evaluated on 1,004 closed, non-pull-request issues from [`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) using strict chronological splitting (80% train / 20% test) to prevent future data leakage.

---

## Group Members

| No. | Full Name | Student ID |
| :---: | :--- | :--- |
| 1 | Samuel Berhanu | ATE/6307/14 |
| 2 | Yordanos Tarekegn | ATE/5260/14 |
| 3 | Henok Zemedkun | ATE/8552/16 |
| 4 | Eyob Kassaye | ATE/4534/16 |
| 5 | Fitsum Yoseph | ATE/5804/16 |

---

## 1. Problem & User Context

- **Target User**: Open-source maintainers triaging dozens of incoming issue reports each week.
- **Problem**: Manual issue inspection and labeling is repetitive and time-consuming. Misclassified bug reports can delay critical fixes, while misfiled usage questions clutter bug trackers.
- **Objective**: Automate classification into 4 standardized labels:
  - `bug`: Code defects, tracebacks, regressions, unexpected crashes.
  - `feature`: Enhancements, new estimator algorithms, API extensions.
  - `question`: Usage inquiries, installation hurdles, configuration help.
  - `docs`: Docstring corrections, tutorial additions, typographical fixes.
- **Dataset**: 1,004 closed, human-labeled issues ingested via GitHub REST API with pull requests strictly excluded (`'pull_request' not in item`).

---

## 2. Architecture & Adaptation Rationale

### Why Encoder-Only Transformers for Classification?
Text classification is an **input text $\to$ discrete label** task. In comparing Transformer model families:
1. **Encoder-Only (BERT, DistilBERT, MiniLM)**: Bidirectional self-attention allows every token to attend to tokens on both its left and right simultaneously. This produces dense, pooled representations of whole documents in a single forward pass without autoregressive generation overhead.
2. **Decoder-Only (e.g. GPT, Llama)**: Causal masking restricts attention to preceding tokens to generate text token-by-token. For fixed-label classification, generating words autoregressively introduces sampling variance, decoding latency, prompt format brittleness, and unnecessary token compute costs.
3. **Encoder-Decoder (e.g. T5, BART)**: Sequence-to-sequence translation models add decoder cross-attention layers and beam search decoding, which is architectural bloat when mapping to 4 discrete categories.

### Frozen Embeddings vs Fine-Tuning
We compared two distinct encoder adaptation paradigms:
- **Frozen Sentence Embeddings (`sentence-transformers/all-MiniLM-L6-v2` + Logistic Regression)**:
  - Generates fixed 384-dimensional dense semantic vectors.
  - The downstream linear classifier is convex and trains in seconds.
  - Embeddings are cached locally (`data/cache/`), enabling ultra-fast inference (0.32 ms/issue) and zero risk of catastrophic forgetting.
- **End-to-End Fine-Tuning (`distilbert-base-uncased`)**:
  - Adds a sequence classification head on top of 6 Transformer encoder layers (66M parameters).
  - Adapts all weights via AdamW ($lr=3\times 10^{-5}$) with linear learning rate warmup.
  - While theoretically capable of learning task-specific representations, fine-tuning on small corpora (802 training issues) is prone to overfitting and incurs ~50 ms/issue inference latency on CPU.

---

## 3. Quickstart (Laptop / Free Colab)

### Step 1: Environment Setup
```bash
# clone repository and enter directory
git clone https://github.com/example/bug-triage.git
cd bug-triage

# create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install required dependencies
pip install -r requirements.txt
```

### Step 2: Environment Variables (Optional for Live API Fetching)
All raw issues and chronological splits are pre-cached in `data/raw/` and `data/processed/`. If you want to re-fetch live data from GitHub, create a `.env` file with your token:
```bash
echo "GITHUB_TOKEN=your_personal_access_token_here" > .env
```

### Step 3: Run Unit Tests
Verify data pipeline integrity, PR filtering, label mapping, and model contracts:
```bash
pytest
```
*Expected output: 6 passed in ~1 second.*

### Step 4: Reproduce Evaluation End-to-End
All pipelines can be run individually using cached data:

```bash
# run baseline 1 (keyword rules) and baseline 2 (tfidf + logistic regression)
python run_milestone2.py

# run model 3 (all-MiniLM-L6-v2 embeddings + logistic regression)
python run_milestone3.py

# run model 4 (distilbert fine-tuning/evaluation) and generate final report
python run_milestone4.py

# run automated failure analysis and error categorization
python src/failure_analysis.py
```

---

## 4. Honest Evaluation & Benchmark Results

### Chronological Split Discipline
- To simulate real-world production conditions, all issues were sorted strictly by `created_at` timestamp.
- The **oldest 80% (802 issues)** form the training set, while the **newest 20% (202 issues)** form the evaluation test set.
- We stratified chronologically per class to prevent historical label distribution shifts (e.g. `question` issues transitioning to GitHub Discussions).
- **Never randomly shuffled**: Random k-fold cross-validation introduces severe temporal data leakage, as future API discussions and version identifiers leak into training.

### Overall Benchmark Summary (Chronological Test Set: 202 Issues)

| Approach | Macro-F1 | Accuracy | Latency (ms/issue) | Est. Cost ($/issue) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Deterministic Keyword Rules** | 0.5554 | 66.8% | 0.46 ms | $0.000000 |
| **Baseline 2: TF-IDF + Logistic Regression** | **0.8044** | **85.1%** | **1.28 ms** | **$0.000001** |
| **Model 3: `all-MiniLM-L6-v2` + Logistic Regression** | 0.6381 | 70.8% | 0.32 ms* | $0.000010 |
| **Model 4: Fine-Tuned DistilBERT (`distilbert-base-uncased`)** | 0.5438 | 66.8% | 52.09 ms | $0.000050 |

*\*Cached embedding inference. Cold-start CPU embedding extraction takes ~41 ms/issue.*

### Per-Class F1 Score Breakdown

| Approach | `bug` F1 | `feature` F1 | `question` F1 | `docs` F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Keyword Rules** | 0.692 | 0.694 | 0.067 | 0.769 |
| **Baseline 2: TF-IDF + Logistic Regression** | **0.794** | **0.922** | **0.588** | **0.914** |
| **Model 3: `all-MiniLM-L6-v2`** | 0.694 | 0.841 | 0.293 | 0.724 |
| **Model 4: Fine-Tuned DistilBERT** | 0.571 | 0.787 | 0.080 | 0.737 |

### Confusion Matrix Summary

#### Baseline 2: TF-IDF + Logistic Regression
```
True \ Pred     bug   feature  question      docs
bug              50         5         0         5
feature           1        59         0         0
question         13         1        10         0
docs              2         3         0        53
```

#### Model 3: `all-MiniLM-L6-v2` + Logistic Regression
```
True \ Pred     bug   feature  question      docs
bug              42         5         2        11
feature           1        53         3         3
question         12         4         6         2
docs              6         4         6        42
```

#### Model 4: Fine-Tuned DistilBERT
```
True \ Pred     bug   feature  question      docs
bug              30         4         0        26
feature           0        48         0        12
question         14         9         1         0
docs              1         1         0        56
```

### Why Did TF-IDF Outperform Deep Neural Encoders?
1. **Template-Driven GitHub Issues**: Technical bug and feature reports adhere to standardized issue templates containing explicit lexical markers (`Traceback (most recent call last)`, `Expected behavior`, `docstring`, `sphinx`). Sparse n-gram models assign direct, unbounded linear weights to these exact discriminative tokens.
2. **Dense Semantic Blurring**: General-domain sentence transformers (`all-MiniLM-L6-v2`) project specific technical terms and code snippets into a smoothed continuous space, diminishing the high-signal impact of rare syntax tokens.
3. **Sample Efficiency & Overfitting**: Training 66M parameters in DistilBERT on 802 samples causes the network to memorize training-specific phrasing rather than generalizable category boundaries.

---

## 5. Failure Analysis

Detailed diagnostic inspection of misclassifications from [`results/failure_analysis.md`](results/failure_analysis.md):

### Quantitative Distribution of Errors

| Failure Category | TF-IDF Errors | TF-IDF % | MiniLM Errors | MiniLM % |
| :--- | :---: | :---: | :---: | :---: |
| **Class Imbalance / Rare Class** (`question`) | 14 | 46.7% | 29 | 49.2% |
| **Ambiguous or Multi-Intent Boundary** (`bug` ↔ `feature`) | 9 | 30.0% | 13 | 22.0% |
| **Cross-Domain Keywords** (`bug` ↔ `docs`) | 7 | 23.3% | 17 | 28.8% |
| **Total Test Errors** | **30 / 202** | **14.9%** | **59 / 202** | **29.2%** |

### Real Failure Examples

1. **Class Imbalance (`question` misclassified as `bug`)**:
   - *Example*: Issue #24584: `"apple silicon m1 can't importing sklearn package."`
   - *True*: `question` | *Predicted*: `bug`
   - *Diagnosis*: Usage question containing terminal error messages (`Symbol not found`, `import error`). Classifiers latch onto error logs and predict `bug`.
2. **Ambiguous Boundary (`bug` vs `feature`)**:
   - *Example*: Issue #25164: `"CalibratedClassifierCV(method='sigmoid') should always take the raw logits as inputs instead of the predicted probabilities"`
   - *True*: `bug` | *Predicted*: `feature`
   - *Diagnosis*: Phrased prescriptively ("should always take..."), matching feature enhancement syntax despite describing an algorithmic oversight.
3. **Cross-Domain Keywords (`bug` vs `docs`)**:
   - *Example*: Issue #24888: `"DOI badge broken in README.rst (Zenodo SVG not rendering on GitHub)"`
   - *True*: `bug` | *Predicted*: `docs`
   - *Diagnosis*: Defect in documentation assets mixes `README.rst` and `SVG` keywords with bug concepts.
4. **Inference Latency & Cost Degradation**:
   - DistilBERT required 52.09 ms/issue (40x slower than TF-IDF) and 50x higher compute cost without accuracy improvement.

---

## 6. Project 2 Readiness: Adaptive Router Integration

Every model implements the [`BaseClassifier`](src/models/base.py) interface:
```python
@dataclass
class PredictionOutput:
    label: str
    latency_ms: float
    cost_usd: float
    confidence: Optional[float] = None
```

In **Project 2**, an adaptive router decides between fast local encoders and expensive LLM calls to optimize utility:
$$\text{Reward} = \mathbb{I}(\hat{y} = y) - \text{cost\_usd} - \lambda \cdot \text{latency}$$

### Routing Strategy
- **Fast-Path (TF-IDF / MiniLM)**: When prediction confidence $\ge 0.85$ (e.g. unambiguous bug reports or doc updates), route immediately to TF-IDF ($0.000001, ~1.3 ms).
- **Slow-Path (LLM Fallback)**: When confidence is marginal ($< 0.85$) or the predicted class is ambiguous (`question` vs `bug`), route to a zero-shot LLM with chain-of-thought verification.
