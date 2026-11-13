import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from src.dataset import prepare_dataset
from src.models.rule_based import RuleBasedClassifier
from src.models.tfidf import TfidfLogisticClassifier
from src.models.embedding import EmbeddingClassifier
from src.models.distilbert import DistilBertClassifier

console = Console()

PRESET_ISSUES = [
    {
        "name": "Traceback bug (ValueError in fit)",
        "label": "bug",
        "title": "ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required.",
        "body": "When passing empty batches to SGDClassifier.partial_fit, the following traceback occurs:\nTraceback (most recent call last):\n  File 'test.py', line 12, in <module>\n    clf.partial_fit(X, y)\nValueError: Found array with 0 sample(s). Expected behavior: should raise empty batch warning."
    },
    {
        "name": "Feature request (GPU acceleration)",
        "label": "feature",
        "title": "ENH: Support cuML or PyTorch backend for TSNE and PCA",
        "body": "It would be great to allow optional GPU acceleration when PyTorch or CuPy is installed. This would speed up high-dimensional manifold learning significantly on large datasets."
    },
    {
        "name": "Usage question (Evaluation parameter)",
        "label": "question",
        "title": "precision_recall_curve with softmax output: which probability column to pass?",
        "body": "Hi, I have a binary classifier outputting shape (n, 2) softmax logits. Which column should I feed to precision_recall_curve when pos_label=0? Is it[:, 0] or [:, 1]? Thanks in advance!"
    },
    {
        "name": "Documentation issue (Broken link in guide)",
        "label": "docs",
        "title": "DOC: Broken cross-reference link in user guide for HistGradientBoosting",
        "body": "The link to early stopping parameter documentation in doc/modules/ensemble.rst points to a 404 page on the stable documentation website. Needs fixing in sphinx markup."
    },
    {
        "name": "Tricky boundary case (UI accessibility defect)",
        "label": "bug",
        "title": "HTML repr: keyboard focus is invisible on expand/collapse toggles (WCAG 2.4.7)",
        "body": "When navigating estimator HTML representation using keyboard tab key, the focus outline is missing. This violates WCAG 2.4.7 focus visible criteria in Jupyter notebook outputs."
    }
]


def load_all_models():
    with console.status("[dim]Loading models and reference data...", spinner="line"):
        t0 = time.time()
        train_df, _ = prepare_dataset()
        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()

        rules = RuleBasedClassifier()

        tfidf = TfidfLogisticClassifier()
        tfidf.fit(train_texts, train_labels)

        minilm = EmbeddingClassifier()
        minilm.fit(train_texts, train_labels, cache_key="train_minilm")

        distilbert = DistilBertClassifier()
        distilbert.fit(train_texts, train_labels)

        load_sec = time.time() - t0
        console.print(f"[dim]Models ready in {load_sec:.1f}s[/dim]\n")
        return [rules, tfidf, minilm, distilbert]


def run_single_demo(models, title: str, body: str, true_label: str = ""):
    full_text = f"{title}\n\n{body}".strip()

    header = f"[bold]Title:[/bold] {title}\n"
    if true_label:
        header += f"[bold]Ground truth:[/bold] [reverse] {true_label} [/reverse]\n"
    header += f"[dim]Body snippet: {body[:140]}...[/dim]"

    console.print(Panel(header, title="[bold]Target issue under triage[/bold]", border_style="grey50"))

    table = Table(title="Multi-model triage comparison", show_header=True, header_style="bold white", border_style="grey37")
    table.add_column("Approach", style="dim", width=28)
    table.add_column("Prediction", justify="center", width=14)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Latency", justify="right", width=14)
    table.add_column("Cost", justify="right", width=13)
    table.add_column("Outcome", justify="center", width=12)

    predictions = []
    for model in models:
        out = model.predict_one(full_text)
        predictions.append(out)

        label_cell = f"[bold]{out.label}[/bold]"
        conf_str = f"{out.confidence * 100:.1f}%" if out.confidence is not None else "100.0%"
        lat_str = f"{out.latency_ms:.2f} ms"
        cost_str = f"${out.cost_usd:.6f}"

        if true_label:
            status = "correct" if out.label == true_label else "[dim]mismatch[/dim]"
        else:
            status = "predicted"

        display_name = {
            "baseline_1_rules": "1. Regex rules (Baseline)",
            "baseline_2_tfidf": "2. TF-IDF + LogReg (Best)",
            "encoder_minilm_l6_v2": "3. MiniLM-L6-v2 (Dense)",
            "distilbert_finetuned": "4. DistilBERT (Fine-tuned)"
        }.get(model.name, model.name)

        table.add_row(display_name, label_cell, conf_str, lat_str, cost_str, status)

    console.print(table)

    # simulate project 2 adaptive routing decision
    tfidf_pred = predictions[1]
    tfidf_conf = tfidf_pred.confidence if tfidf_pred.confidence is not None else 0.0

    if tfidf_conf >= 0.80:
        action = (
            f"Fast-path accepted: TF-IDF confidence is [bold]{tfidf_conf*100:.1f}%[/bold] (>= 80.0%).\n"
            f"Classified as [bold]{tfidf_pred.label}[/bold] in {tfidf_pred.latency_ms:.2f} ms (${tfidf_pred.cost_usd:.6f} compute cost).\n"
            f"[dim]Avoided external LLM invocation • Saved ~$0.03 and ~800 ms per issue[/dim]"
        )
    else:
        action = (
            f"Escalation triggered: Local model confidence is marginal ([bold]{tfidf_conf*100:.1f}%[/bold] < 80.0%).\n"
            f"Class boundary ambiguity detected between target classes.\n"
            f"[dim]Routed to downstream reasoning agent for zero-shot verification[/dim]"
        )

    console.print(Panel(action, title="[bold]Project 2 adaptive router decision[/bold]", border_style="grey50"))
    console.print()


def main():
    console.clear()
    console.print(Panel.fit(
        "[bold]Bug-Report Triage: Multi-Model Demonstration[/bold]\n"
        "[dim]Benchmarking Regex Rules, TF-IDF, MiniLM-L6-v2, and DistilBERT with adaptive routing[/dim]",
        border_style="grey50"
    ))

    models = load_all_models()

    while True:
        console.print("[bold]Select an action:[/bold]")
        for i, preset in enumerate(PRESET_ISSUES, start=1):
            console.print(f"  [{i}] Preset: {preset['name']} ({preset['label']})")
        console.print("  [c] Custom issue (title + body)")
        console.print("  [q] Quit")

        choice = Prompt.ask("\nChoice", default="1").strip().lower()

        if choice == "q":
            console.print("[dim]Exited.[/dim]")
            break
        elif choice == "c":
            title = Prompt.ask("Issue title")
            body = Prompt.ask("Issue body")
            run_single_demo(models, title, body)
        elif choice.isdigit() and 1 <= int(choice) <= len(PRESET_ISSUES):
            p = PRESET_ISSUES[int(choice) - 1]
            run_single_demo(models, p["title"], p["body"], p["label"])
        else:
            console.print("[dim]Invalid selection.[/dim]\n")


if __name__ == "__main__":
    main()
