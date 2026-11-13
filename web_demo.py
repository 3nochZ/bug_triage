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
  <title>Bug-Report Triage: Live Multi-Model Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --bug: #ef4444;
      --feature: #22c55e;
      --question: #eab308;
      --docs: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 2rem 1rem; line-height: 1.5; }
    .container { max-width: 1000px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2rem; }
    header h1 { font-size: 2rem; color: var(--primary); margin-bottom: 0.5rem; }
    header p { color: var(--text-muted); }

    .presets { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; justify-content: center; }
    .preset-btn { background: var(--card-bg); border: 1px solid var(--border); color: var(--text); padding: 0.5rem 0.9rem; border-radius: 8px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }
    .preset-btn:hover { border-color: var(--primary); color: var(--primary); }

    .form-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
    .form-group { margin-bottom: 1rem; }
    label { display: block; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.4rem; }
    input[type="text"], textarea { width: 100%; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; color: var(--text); font-size: 0.95rem; }
    input[type="text"]:focus, textarea:focus { outline: none; border-color: var(--primary); }
    textarea { min-height: 120px; resize: vertical; }
    .submit-btn { width: 100%; background: #0284c7; color: white; border: none; padding: 0.85rem; border-radius: 8px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
    .submit-btn:hover { background: #0369a1; }

    .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .model-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
    .model-name { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; margin-bottom: 1rem; }
    .badge.bug { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge.feature { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge.question { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge.docs { background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); }

    .stat-row { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem; color: var(--text-muted); }
    .stat-val { color: var(--text); font-weight: 600; }

    .router-card { background: #132e2b; border: 1px solid #14b8a6; border-radius: 10px; padding: 1.25rem; }
    .router-title { color: #2dd4bf; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
    .router-body { font-size: 0.9rem; color: #ccfbf1; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Bug-Report Triage: Live Multi-Model Demo</h1>
      <p>Multi-Model Comparison: Regex Baseline, TF-IDF + LogReg, MiniLM Dense Embeddings, and DistilBERT</p>
    </header>

    <div class="presets">
      <button class="preset-btn" onclick="setPreset(0)">Preset 1: Traceback Bug</button>
      <button class="preset-btn" onclick="setPreset(1)">Preset 2: GPU Feature Request</button>
      <button class="preset-btn" onclick="setPreset(2)">Preset 3: Usage Question</button>
      <button class="preset-btn" onclick="setPreset(3)">Preset 4: Documentation Link 404</button>
      <button class="preset-btn" onclick="setPreset(4)">Preset 5: Tricky WCAG Defect</button>
    </div>

    <div class="form-card">
      <form id="triage-form" onsubmit="handleTriage(event)">
        <div class="form-group">
          <label>Issue Title</label>
          <input type="text" id="title" required placeholder="e.g. ValueError: Found array with 0 sample(s)...">
        </div>
        <div class="form-group">
          <label>Issue Body / Description</label>
          <textarea id="body" required placeholder="Paste full error logs, tracebacks, or feature description..."></textarea>
        </div>
        <button type="submit" class="submit-btn" id="submit-btn">Run Multi-Model Triage</button>
      </form>
    </div>

    <div id="results-section" style="display: none;">
      <div class="results-grid" id="results-grid"></div>
      <div class="router-card" id="router-card">
        <div class="router-title">⚡ Project 2 Adaptive Router Decision</div>
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
      btn.textContent = 'Running Inference across all 4 models...';

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
        btn.textContent = 'Run Multi-Model Triage';
      }
    }

    function renderResults(data) {
      const grid = document.getElementById('results-grid');
      grid.innerHTML = '';

      const modelMeta = [
        { key: 'rules', name: 'Baseline 1: Rules' },
        { key: 'tfidf', name: 'Baseline 2: TF-IDF (Best)' },
        { key: 'minilm', name: 'Model 3: MiniLM-L6-v2' },
        { key: 'distilbert', name: 'Model 4: DistilBERT' }
      ];

      modelMeta.forEach(m => {
        const p = data[m.key];
        const card = document.createElement('div');
        card.className = 'model-card';
        card.innerHTML = `
          <div class="model-name">${m.name}</div>
          <div><span class="badge ${p.label}">${p.label}</span></div>
          <div class="stat-row"><span>Confidence</span><span class="stat-val">${(p.confidence * 100).toFixed(1)}%</span></div>
          <div class="stat-row"><span>Latency</span><span class="stat-val">${p.latency_ms.toFixed(2)} ms</span></div>
          <div class="stat-row"><span>Est. Cost</span><span class="stat-val">$${p.cost_usd.toFixed(6)}</span></div>
        `;
        grid.appendChild(card);
      });

      const tfidf = data.tfidf;
      const routerText = document.getElementById('router-text');
      if (tfidf.confidence >= 0.80) {
        routerText.innerHTML = `<strong>FAST-PATH ACCEPTED:</strong> Local TF-IDF model returned <strong>${(tfidf.confidence * 100).toFixed(1)}%</strong> confidence.<br>Issue triaged as <strong>${tfidf.label.toUpperCase()}</strong> in <strong>${tfidf.latency_ms.toFixed(2)} ms</strong> at <strong>$${tfidf.cost_usd.toFixed(6)}</strong> compute cost.<br><small style="color: #5eead4;">Project 2 router saves ~$0.03 and 800ms compared to an external LLM call.</small>`;
      } else {
        routerText.innerHTML = `<strong>ESCALATION TO LLM:</strong> TF-IDF confidence is marginal (<strong>${(tfidf.confidence * 100).toFixed(1)}%</strong> < 80%).<br>Potential edge case or ambiguous intent detected. The Project 2 router triggers a fallback reasoning agent for zero-shot classification.`;
      }

      document.getElementById('results-section').style.display = 'block';
      document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
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
