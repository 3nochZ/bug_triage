import sys
import time
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import prepare_dataset
from src.models.rule_based import RuleBasedClassifier
from src.models.tfidf import TfidfLogisticClassifier
from src.models.embedding import EmbeddingClassifier
from src.models.distilbert import DistilBertClassifier

app = Flask(__name__)

# load and prepare models once at server startup
print("loading all 4 models for web demo...")
t0 = time.time()
train_df, _ = prepare_dataset()
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

models = {
    "rules": RuleBasedClassifier(),
    "tfidf": TfidfLogisticClassifier(),
    "minilm": EmbeddingClassifier(),
    "distilbert": DistilBertClassifier()
}

models["tfidf"].fit(train_texts, train_labels)
models["minilm"].fit(train_texts, train_labels, cache_key="train_minilm")
models["distilbert"].fit(train_texts, train_labels)
print(f"all models loaded in {time.time() - t0:.2f}s!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bug-Report Triage</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #09090b;
      --surface: #121215;
      --raised: #18181b;
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.2);
      --border-focus: rgba(255, 255, 255, 0.4);
      --text-primary: #f4f4f5;
      --text-secondary: #a1a1aa;
      --text-muted: #71717a;
      --accent: #f4f4f5;
      --accent-text: #09090b;
      --tag-bg: #222226;
      --tag-border: rgba(255, 255, 255, 0.12);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg);
      color: var(--text-primary);
      font-family: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      padding: 48px 24px;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 960px;
      margin: 0 auto;
    }

    header {
      margin-bottom: 32px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
    }

    .header-meta {
      font-family: "Geist Mono", monospace;
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }

    header h1 {
      font-size: 20px;
      font-weight: 500;
      color: var(--text-primary);
      letter-spacing: -0.01em;
      margin-bottom: 6px;
    }

    header p {
      color: var(--text-secondary);
      font-size: 13px;
    }

    .section-label {
      font-family: "Geist Mono", monospace;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: block;
    }

    .presets-wrap {
      margin-bottom: 24px;
    }

    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .preset-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-family: "Geist", sans-serif;
      font-weight: 400;
      transition: border-color 0.15s, color 0.15s;
    }

    .preset-btn:hover {
      border-color: var(--border-hover);
      color: var(--text-primary);
    }

    .form-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
    }

    .form-group {
      margin-bottom: 16px;
    }

    label {
      display: block;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }

    input[type="text"], textarea {
      width: 100%;
      background: var(--raised);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      color: var(--text-primary);
      font-size: 13px;
      font-family: inherit;
      transition: border-color 0.15s;
    }

    input[type="text"]:focus, textarea:focus {
      outline: none;
      border-color: var(--border-focus);
    }

    textarea {
      min-height: 110px;
      resize: vertical;
      font-family: "Geist Mono", monospace;
      font-size: 12px;
      line-height: 1.6;
    }

    .form-actions {
      display: flex;
      justify-content: flex-end;
    }

    .submit-btn {
      background: var(--accent);
      color: var(--accent-text);
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
      transition: opacity 0.15s;
    }

    .submit-btn:hover {
      opacity: 0.9;
    }

    .submit-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .results-section {
      margin-top: 32px;
      animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }

    @media (max-width: 768px) {
      .results-grid {
        grid-template-columns: 1fr;
      }
    }

    .model-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .model-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
    }

    .model-name {
      font-size: 12px;
      color: var(--text-secondary);
      font-weight: 500;
      line-height: 1.3;
    }

    .model-tag {
      font-family: "Geist Mono", monospace;
      font-size: 11px;
      color: var(--text-muted);
    }

    .prediction-badge-wrap {
      margin-bottom: 20px;
    }

    .badge {
      display: inline-block;
      font-family: "Geist Mono", monospace;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-primary);
      background: var(--tag-bg);
      border: 1px solid var(--tag-border);
      padding: 4px 10px;
      border-radius: 4px;
      letter-spacing: 0.02em;
    }

    .stats-list {
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }

    .stat-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 4px;
      color: var(--text-muted);
    }

    .stat-row:last-child {
      margin-bottom: 0;
    }

    .stat-val {
      font-family: "Geist Mono", monospace;
      color: var(--text-secondary);
      font-variant-numeric: tabular-nums;
    }

    .router-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
    }

    .router-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }

    .router-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-primary);
    }

    .router-title {
      font-family: "Geist Mono", monospace;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-primary);
      font-weight: 500;
    }

    .router-body {
      font-size: 13px;
      color: var(--text-secondary);
      line-height: 1.6;
    }

    .router-body strong {
      color: var(--text-primary);
      font-weight: 500;
    }

    .router-meta {
      margin-top: 6px;
      font-family: "Geist Mono", monospace;
      font-size: 12px;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-meta">Scenario 1 • Encoder Triage</div>
      <h1>Bug-Report Triage System</h1>
      <p>Benchmark inference and adaptive routing evaluation across scikit-learn issues</p>
    </header>

    <div class="presets-wrap">
      <span class="section-label">Preset test scenarios</span>
      <div class="presets">
        <button class="preset-btn" onclick="setPreset(0)">Traceback bug</button>
        <button class="preset-btn" onclick="setPreset(1)">Feature request</button>
        <button class="preset-btn" onclick="setPreset(2)">Usage inquiry</button>
        <button class="preset-btn" onclick="setPreset(3)">Documentation 404</button>
        <button class="preset-btn" onclick="setPreset(4)">Edge defect (WCAG)</button>
      </div>
    </div>

    <div class="form-card">
      <form id="triage-form" onsubmit="handleTriage(event)">
        <div class="form-group">
          <label for="title">Issue title</label>
          <input type="text" id="title" required placeholder="ValueError: Found array with 0 sample(s)...">
        </div>
        <div class="form-group">
          <label for="body">Issue body</label>
          <textarea id="body" required placeholder="Paste traceback, expected behavior, or issue text..."></textarea>
        </div>
        <div class="form-actions">
          <button type="submit" class="submit-btn" id="submit-btn">Run triage</button>
        </div>
      </form>
    </div>

    <div id="results-section" class="results-section" style="display: none;">
      <span class="section-label">Model predictions</span>
      <div class="results-grid" id="results-grid"></div>
      <div class="router-card">
        <div class="router-header">
          <div class="router-dot"></div>
          <div class="router-title">Project 2 router policy</div>
        </div>
        <div class="router-body" id="router-text"></div>
      </div>
    </div>
  </div>

  <script>
    const presets = [
      {
        title: "ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required.",
        body: "When passing empty batches to SGDClassifier.partial_fit, the following traceback occurs:\\nTraceback (most recent call last):\\n  File 'test.py', line 12, in <module>\\n    clf.partial_fit(X, y)\\nValueError: Found array with 0 sample(s). Expected behavior: should raise empty batch warning."
      },
      {
        title: "ENH: Support cuML or PyTorch backend for TSNE and PCA",
        body: "It would be great to allow optional GPU acceleration when PyTorch or CuPy is installed. This would speed up high-dimensional manifold learning significantly on large datasets."
      },
      {
        title: "precision_recall_curve with softmax output: which probability column to pass?",
        body: "Hi, I have a binary classifier outputting shape (n, 2) softmax logits. Which column should I feed to precision_recall_curve when pos_label=0? Is it[:, 0] or [:, 1]? Thanks in advance!"
      },
      {
        title: "DOC: Broken cross-reference link in user guide for HistGradientBoosting",
        body: "The link to early stopping parameter documentation in doc/modules/ensemble.rst points to a 404 page on the stable documentation website. Needs fixing in sphinx markup."
      },
      {
        title: "HTML repr: keyboard focus is invisible on expand/collapse toggles (WCAG 2.4.7)",
        body: "When navigating estimator HTML representation using keyboard tab key, the focus outline is missing. This violates WCAG 2.4.7 focus visible criteria in Jupyter notebook outputs."
      }
    ];

    function setPreset(idx) {
      document.getElementById('title').value = presets[idx].title;
      document.getElementById('body').value = presets[idx].body;
      document.getElementById('submit-btn').click();
    }

    async function handleTriage(e) {
      e.preventDefault();
      const title = document.getElementById('title').value;
      const body = document.getElementById('body').value;
      const btn = document.getElementById('submit-btn');

      btn.disabled = true;
      btn.textContent = 'Evaluating...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, body })
        });
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert('Inference error: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Run triage';
      }
    }

    function renderResults(data) {
      const grid = document.getElementById('results-grid');
      grid.innerHTML = '';

      const modelMeta = [
        { key: 'rules', name: 'Regex rules', tag: 'Baseline 1' },
        { key: 'tfidf', name: 'TF-IDF + LogReg', tag: 'Baseline 2' },
        { key: 'minilm', name: 'MiniLM-L6-v2', tag: 'Model 3' },
        { key: 'distilbert', name: 'DistilBERT', tag: 'Model 4' }
      ];

      modelMeta.forEach(m => {
        const p = data[m.key];
        const card = document.createElement('div');
        card.className = 'model-card';
        card.innerHTML = `
          <div>
            <div class="model-header">
              <span class="model-name">${m.name}</span>
              <span class="model-tag">${m.tag}</span>
            </div>
            <div class="prediction-badge-wrap">
              <span class="badge">${p.label}</span>
            </div>
          </div>
          <div class="stats-list">
            <div class="stat-row">
              <span>Confidence</span>
              <span class="stat-val">${(p.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="stat-row">
              <span>Latency</span>
              <span class="stat-val">${p.latency_ms.toFixed(2)} ms</span>
            </div>
            <div class="stat-row">
              <span>Cost</span>
              <span class="stat-val">$${p.cost_usd.toFixed(6)}</span>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });

      const tfidf = data.tfidf;
      const routerText = document.getElementById('router-text');
      if (tfidf.confidence >= 0.80) {
        routerText.innerHTML = `
          Fast-path routing accepted. Local TF-IDF model returned <strong>${(tfidf.confidence * 100).toFixed(1)}%</strong> confidence.<br>
          Issue triaged as <strong>${tfidf.label}</strong> in <strong>${tfidf.latency_ms.toFixed(2)} ms</strong> at <strong>$${tfidf.cost_usd.toFixed(6)}</strong> compute cost.
          <div class="router-meta">Avoided external LLM invocation • Saved ~$0.03 and 800ms per issue</div>
        `;
      } else {
        routerText.innerHTML = `
          Escalation triggered. Primary model confidence is marginal (<strong>${(tfidf.confidence * 100).toFixed(1)}%</strong> < 80.0%).<br>
          Potential boundary ambiguity detected between target classes.
          <div class="router-meta">Routed to downstream reasoning agent for zero-shot verification</div>
        `;
      }

      document.getElementById('results-section').style.display = 'block';
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    title = data.get("title", "")
    body = data.get("body", "")
    full_text = f"{title}\n\n{body}".strip()

    results = {}
    for key, model in models.items():
        out = model.predict_one(full_text)
        results[key] = {
            "label": out.label,
            "latency_ms": out.latency_ms,
            "cost_usd": out.cost_usd,
            "confidence": out.confidence if out.confidence is not None else 1.0
        }

    return jsonify(results)


if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f" Live Web Demo Running at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
