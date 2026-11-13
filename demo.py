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
        "name": "Traceback Bug (ValueError in fit)",
        "label": "bug",
        "title": "ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required.",
        "body": "When passing empty batches to SGDClassifier.partial_fit, the following traceback occurs:\nTraceback (most recent call last):\n  File 'test.py', line 12, in <module>\n    clf.partial_fit(X, y)\nValueError: Found array with 0 sample(s). Expected behavior: should raise empty batch warning."
    },
    {
        "name": "Feature Request (GPU acceleration)",
        "label": "feature",
        "title": "ENH: Support cuML or PyTorch backend for TSNE and PCA",
        "body": "It would be great to allow optional GPU acceleration when PyTorch or CuPy is installed. This would speed up high-dimensional manifold learning significantly on large datasets."
    },
    {
        "name": "Usage Question (Evaluation parameter)",
        "label": "question",
        "title": "precision_recall_curve with softmax output: which probability column to pass?",
        "body": "Hi, I have a binary classifier outputting shape (n, 2) softmax logits. Which column should I feed to precision_recall_curve when pos_label=0? Is it[:, 0] or [:, 1]? Thanks in advance!"
    },
    {
        "name": "Documentation Issue (Broken link in guide)",
        "label": "docs",
        "title": "DOC: Broken cross-reference link in user guide for HistGradientBoosting",
        "body": "The link to early stopping parameter documentation in doc/modules/ensemble.rst points to a 404 page on the stable documentation website. Needs fixing in sphinx markup."
    },
    {
        "name": "Tricky Boundary Case (UI accessibility defect)",
        "label": "bug",
        "title": "HTML repr: keyboard focus is invisible on expand/collapse toggles (WCAG 2.4.7)",
        "body": "When navigating estimator HTML representation using keyboard tab key, the focus outline is missing. This violates WCAG 2.4.7 focus visible criteria in Jupyter notebook outputs."
    }
]

COLOR_MAP = {
    "bug": "bold red",
    "feature": "bold green",
    "question": "bold yellow",
    "docs": "bold cyan"
}


def load_all_models():
    with console.status("[bold cyan]Loading models and training data...", spinner="dots"):
        t0 = time.time()
        train_df, _ = prepare_dataset()
        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()

        # 1. rule-based
        rules = RuleBasedClassifier()

        # 2. tf-idf + logreg
        tfidf = TfidfLogisticClassifier()
        tfidf.fit(train_texts, train_labels)

        # 3. minilm embeddings + logreg
        minilm = EmbeddingClassifier()
        minilm.fit(train_texts, train_labels, cache_key="train_minilm")

        # 4. distilbert fine-tuned
        distilbert = DistilBertClassifier()
        distilbert.fit(train_texts, train_labels)

        load_sec = time.time() - t0
        console.print(f"[bold green]✓[/bold green] All 4 models loaded and ready in [bold]{load_sec:.1f}s[/bold]!\n")
        return [rules, tfidf, minilm, distilbert]


def run_single_demo(models, title: str, body: str, true_label: str = ""):
    full_text = f"{title}\n\n{body}".strip()

    header = f"[bold white]Title:[/bold white] {title}\n"
    if true_label:
        badge = f"[{COLOR_MAP.get(true_label, 'white')}]{true_label.upper()}[/{COLOR_MAP.get(true_label, 'white')}]"
        header += f"[bold white]Ground Truth Label:[/bold white] {badge}\n"
    header += f"[bold white]Body Preview:[/bold white] {body[:140]}..."

    console.print(Panel(header, title="[bold yellow]Target Issue Under Triage[/bold yellow]", border_style="blue"))

    table = Table(title="Live Multi-Model Triage Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Model Approach", style="dim", width=28)
    table.add_column("Predicted Class", justify="center", width=16)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Est. Cost ($)", justify="right", width=13)
    table.add_column("Status", justify="center", width=10)

    predictions = []
    for model in models:
        out = model.predict_one(full_text)
        predictions.append(out)

        color = COLOR_MAP.get(out.label, "white")
        label_cell = f"[{color}]{out.label.upper()}[/{color}]"
        conf_str = f"{out.confidence * 100:.1f}%" if out.confidence is not None else "100.0%"
        lat_str = f"{out.latency_ms:.2f} ms"
        cost_str = f"${out.cost_usd:.6f}"

        if true_label:
            status = "[green]✓ CORRECT[/green]" if out.label == true_label else "[red]✗ ERROR[/red]"
        else:
            status = "[cyan]PREDICTED[/cyan]"

        display_name = {
            "baseline_1_rules": "1. Keyword Rules (Baseline)",
            "baseline_2_tfidf": "2. TF-IDF + LogReg (Best)",
            "encoder_minilm_l6_v2": "3. all-MiniLM-L6-v2 (Dense)",
            "distilbert_finetuned": "4. DistilBERT (Fine-Tuned)"
        }.get(model.name, model.name)

        table.add_row(display_name, label_cell, conf_str, lat_str, cost_str, status)

    console.print(table)

    # simulate project 2 adaptive routing decision
    tfidf_pred = predictions[1]
    tfidf_conf = tfidf_pred.confidence if tfidf_pred.confidence is not None else 0.0

    if tfidf_conf >= 0.80:
        action = (
            f"[bold green]FAST-PATH ROUTED[/bold green]: TF-IDF confidence is [bold]{tfidf_conf*100:.1f}%[/bold] (>= 80%).\n"
            f"Triaged as [{COLOR_MAP[tfidf_pred.label]}]{tfidf_pred.label.upper()}[/{COLOR_MAP[tfidf_pred.label]}] in "
            f"[bold]{tfidf_pred.latency_ms:.2f}ms[/bold] with [bold]${tfidf_pred.cost_usd:.6f}[/bold] compute cost.\n"
            f"[dim]Avoided costly external LLM API invocation (saved ~$0.02 - $0.05 and ~800ms).[/dim]"
        )
    else:
        action = (
            f"[bold yellow]ROUTER ESCALATION[/bold yellow]: Local model confidence is low ([bold]{tfidf_conf*100:.1f}%[/bold] < 80%).\n"
            f"Edge case detected! Project 2 router escalates issue to [bold cyan]LLM Reasoning Agent[/bold cyan] for zero-shot clarification."
        )

    console.print(Panel(action, title="[bold green]Project 2 Adaptive Router Decision[/bold green]", border_style="green"))
    console.print()


def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Bug-Report Triage: Live Multi-Model Demonstration[/bold cyan]\n"
        "[dim]Comparing Regex Rules, TF-IDF, MiniLM-L6-v2, and DistilBERT with Project 2 Router[/dim]",
        border_style="cyan"
    ))

    models = load_all_models()

    while True:
        console.print("[bold white]Choose an action:[/bold white]")
        for i, preset in enumerate(PRESET_ISSUES, start=1):
            console.print(f"  [[bold cyan]{i}[/bold cyan]] Demo Preset: {preset['name']} ({preset['label']})")
        console.print("  [[bold cyan]c[/bold cyan]] Enter Custom Issue (Title + Body)")
        console.print("  [[bold cyan]q[/bold cyan]] Quit")

        choice = Prompt.ask("\nSelect option", default="1").strip().lower()

        if choice == "q":
            console.print("[bold green]Demo finished. Goodbye![/bold green]")
            break
        elif choice == "c":
            title = Prompt.ask("Issue Title")
            body = Prompt.ask("Issue Body")
            run_single_demo(models, title, body)
        elif choice.isdigit() and 1 <= int(choice) <= len(PRESET_ISSUES):
            p = PRESET_ISSUES[int(choice) - 1]
            run_single_demo(models, p["title"], p["body"], p["label"])
        else:
            console.print("[red]Invalid selection. Try again.[/red]\n")


if __name__ == "__main__":
    main()
