import json
import os
import time
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_pennylane_data_reuploading import PennyLaneDataReuploadingProcess

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.html")

CONFIGS = [
    {
        "id": "baseline",
        "title": "Baseline Classifier",
        "subtitle": "3 layers, 5 epochs",
        "description": (
            "Standard data-reuploading quantum classifier using a 3-layer "
            "single-qubit circuit. Trains on 50 circle-dataset points for 5 "
            "epochs with Adam optimizer."
        ),
        "config": {
            "num_layers": 3,
            "learning_rate": 0.6,
            "epochs": 5,
            "batch_size": 16,
            "num_train": 50,
            "num_test": 100,
            "seed": 42,
        },
        "accent": "#636efa",
        "dataset": {
            "name": "Circle 2D",
            "source": "Inline generation (radius = sqrt(2/pi))",
            "classes": 2,
        },
    },
    {
        "id": "deep",
        "title": "Deep Re-uploading",
        "subtitle": "5 layers, 5 epochs",
        "description": (
            "Deeper variant with 5 re-uploading layers to test whether "
            "additional circuit depth improves classification accuracy."
        ),
        "config": {
            "num_layers": 5,
            "learning_rate": 0.4,
            "epochs": 5,
            "batch_size": 16,
            "num_train": 50,
            "num_test": 100,
            "seed": 42,
        },
        "accent": "#ef553b",
        "dataset": {
            "name": "Circle 2D",
            "source": "Inline generation",
            "classes": 2,
        },
    },
    {
        "id": "fast",
        "title": "Fast Smoke Test",
        "subtitle": "2 layers, 3 epochs",
        "description": (
            "Minimal configuration for rapid validation: 2 layers, 3 epochs, "
            "30 training samples. Useful for CI and pipeline smoke tests."
        ),
        "config": {
            "num_layers": 2,
            "learning_rate": 0.6,
            "epochs": 3,
            "batch_size": 16,
            "num_train": 30,
            "num_test": 50,
            "seed": 42,
        },
        "accent": "#00cc96",
        "dataset": {
            "name": "Circle 2D",
            "source": "Inline generation",
            "classes": 2,
        },
    },
]


def build_core():
    core = allocate_core()
    core.register_link("ram-emitter", RAMEmitter)
    core.register_link("PennyLaneDataReuploadingProcess", PennyLaneDataReuploadingProcess)
    return core


def run_config(cfg):
    c = cfg["config"]
    core = build_core()
    proc = PennyLaneDataReuploadingProcess(config=c, core=core)
    state = proc.initial_state()

    history = []
    t0 = time.perf_counter()
    for _ in range(c["epochs"] + 2):
        result = proc.update(state, interval=1.0)
        state = {**state, **result}
        entry = {
            "phase": result["phase"],
            "epoch": result["epoch"],
            "loss": result.get("loss", 0.0),
            "train_accuracy": result.get("train_accuracy", 0.0),
            "test_accuracy": result.get("test_accuracy", 0.0),
            "best_test_accuracy": result.get("best_test_accuracy", 0.0),
        }
        history.append(entry)
        if result["phase"] == "done":
            break
    elapsed = time.perf_counter() - t0

    return {"history": history, "elapsed": elapsed, "final": history[-1] if history else {}}


def make_metrics_card(title, value, subtitle, accent):
    return f"""
    <div class="metric-card" style="border-left: 4px solid {accent};">
        <div class="metric-value">{value}</div>
        <div class="metric-title">{title}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>"""


def make_time_series_chart(histories, accent):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Accuracy", "Loss"),
    )
    for label, hist, color in histories:
        epochs = [h["epoch"] for h in hist if h["phase"] != "init"]
        train_acc = [h["train_accuracy"] for h in hist if h["phase"] != "init"]
        test_acc = [h["test_accuracy"] for h in hist if h["phase"] != "init"]
        losses = [h["loss"] for h in hist if h["phase"] != "init"]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_acc,
                mode="lines+markers",
                name=f"{label} (train)",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=test_acc,
                mode="lines+markers",
                name=f"{label} (test)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def make_pbg_document_viewer(doc):
    html = json.dumps(doc, indent=2)
    html = (
        html.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f"""
    <div class="pbg-doc-viewer">
        <pre><code>{html}</code></pre>
    </div>"""


def make_bigraph_diagram(doc, accent, title):
    """Render a bigraph-viz2 interactive diagram fragment."""
    from bigraph_viz2 import emit_html

    snippet = emit_html(doc, height="420px", inspector=True, dedupe=False)
    return snippet


def generate_report():
    results = {}
    for cfg in CONFIGS:
        print(f"  Running {cfg['id']}...")
        t0 = time.time()
        results[cfg["id"]] = run_config(cfg)
        dt = time.time() - t0
        final = results[cfg["id"]]["final"]
        print(
            f"    Done in {dt:.1f}s | "
            f"final train_acc={final.get('train_accuracy', 0):.3f} "
            f"test_acc={final.get('test_accuracy', 0):.3f}"
        )

    sections_html = ""
    for i, cfg in enumerate(CONFIGS):
        result = results[cfg["id"]]
        final = result["final"]
        hist = result["history"]

        metrics = ""
        metrics += make_metrics_card(
            "Final Test Accuracy",
            f"{final.get('test_accuracy', 0):.1%}",
            f"Best: {final.get('best_test_accuracy', 0):.1%}",
            cfg["accent"],
        )
        metrics += make_metrics_card(
            "Final Train Accuracy",
            f"{final.get('train_accuracy', 0):.1%}",
            f"Epochs: {final.get('epoch', 0)}",
            cfg["accent"],
        )
        metrics += make_metrics_card(
            "Final Loss",
            f"{final.get('loss', 0):.4f}",
            "Fidelity-based cost",
            cfg["accent"],
        )
        metrics += make_metrics_card(
            "Wall Clock",
            f"{result['elapsed']:.1f}s",
            f"{cfg['config']['epochs']} epochs",
            cfg["accent"],
        )

        chart_html = make_time_series_chart(
            [(cfg["title"], hist, cfg["accent"])],
            cfg["accent"],
        )

        doc = {
            "classifier": {
                "_type": "process",
                "address": "local:PennyLaneDataReuploadingProcess",
                "config": cfg["config"],
                "inputs": {},
                "outputs": {
                    "phase": {"_type": "string"},
                    "epoch": {"_type": "integer"},
                    "loss": {"_type": "float"},
                    "train_accuracy": {"_type": "float"},
                    "test_accuracy": {"_type": "float"},
                    "best_test_accuracy": {"_type": "float"},
                },
            },
            "stores": {
                "phase": "string",
                "epoch": "integer",
                "loss": "float",
                "train_accuracy": "float",
                "test_accuracy": "float",
                "best_test_accuracy": "float",
            },
        }
        bigraph_html = make_bigraph_diagram(doc, cfg["accent"], cfg["title"])
        doc_viewer = make_pbg_document_viewer(doc)

        section_id = f"section-{cfg['id']}"
        sections_html += f"""
        <section id="{section_id}" class="config-section">
            <h2 style="border-left: 4px solid {cfg['accent']}; padding-left: 12px;">
                {cfg['title']}
                <span class="subtitle">{cfg['subtitle']}</span>
            </h2>
            <p class="description">{cfg['description']}</p>
            <div class="metrics-grid">{metrics}</div>
            <div class="chart-container">{chart_html}</div>
            <div class="two-col">
                <div class="col">
                    <h3>Bigraph Architecture</h3>
                    {bigraph_html}
                </div>
                <div class="col">
                    <h3>PBG Document</h3>
                    {doc_viewer}
                </div>
            </div>
        </section>"""

    nav_items = "".join(
        f'<a href="#section-{cfg["id"]}" style="color: {cfg["accent"]};">{cfg["title"]}</a>'
        for cfg in CONFIGS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-pennylane-data-reuploading — Demo Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f8f9fa; color: #333; line-height: 1.6;
}}
nav {{
    position: sticky; top: 0; z-index: 100;
    background: #fff; border-bottom: 1px solid #e0e0e0;
    padding: 12px 24px; display: flex; align-items: center; gap: 20px;
    flex-wrap: wrap;
}}
nav .brand {{ font-weight: 700; font-size: 1.1rem; color: #333; }}
nav a {{ text-decoration: none; font-size: 0.9rem; font-weight: 500; }}
nav a:hover {{ text-decoration: underline; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
.hero {{
    background: linear-gradient(135deg, #636efa 0%, #7c3aed 100%);
    color: #fff; padding: 48px 32px; border-radius: 12px;
    margin-bottom: 32px;
}}
.hero h1 {{ font-size: 2rem; margin-bottom: 8px; }}
.hero p {{ opacity: 0.9; font-size: 1rem; }}
.hero .meta {{ margin-top: 16px; font-size: 0.85rem; opacity: 0.75; }}
.metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin: 20px 0;
}}
.metric-card {{
    background: #fff; border-radius: 8px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.metric-value {{ font-size: 1.6rem; font-weight: 700; color: #333; }}
.metric-title {{ font-size: 0.8rem; color: #666; margin-top: 4px; }}
.metric-subtitle {{ font-size: 0.75rem; color: #999; }}
.chart-container {{ background: #fff; border-radius: 8px; padding: 16px; margin: 20px 0; }}
.config-section {{ margin-bottom: 48px; }}
.config-section h2 {{ font-size: 1.4rem; margin-bottom: 8px; }}
.config-section .subtitle {{ font-size: 0.85rem; color: #666; font-weight: 400; margin-left: 8px; }}
.config-section .description {{ color: #555; margin-bottom: 12px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 20px; }}
.two-col .col {{ background: #fff; border-radius: 8px; padding: 16px; }}
.two-col .col h3 {{ font-size: 1rem; margin-bottom: 12px; color: #555; }}
.pbg-doc-viewer {{
    background: #1e1e2e; border-radius: 6px; padding: 12px;
    max-height: 420px; overflow: auto;
}}
.pbg-doc-viewer code {{
    font-family: "SF Mono", "Fira Code", "Fira Mono", monospace;
    font-size: 0.75rem; color: #cdd6f4; white-space: pre;
}}
footer {{ text-align: center; padding: 24px; color: #999; font-size: 0.8rem; }}
</style>
</head>
<body>
<nav>
    <span class="brand">pbg-pennylane-data-reuploading</span>
    {nav_items}
    <a href="https://pennylane.ai/demos/tutorial_data_reuploading_classifier" target="_blank" style="margin-left: auto; color: #666;">PennyLane Demo</a>
</nav>
<div class="container">
    <div class="hero">
        <h1>Data-Reuploading Quantum Classifier</h1>
        <p>Process-bigraph wrapper for PennyLane's single-qubit universal quantum classifier (Pérez-Salinas et al. 2019)</p>
        <div class="meta">
            Generated: {time.strftime("%Y-%m-%d %H:%M:%S")} &middot;
            PenaltyLane 0.45.1 &middot;
            process-bigraph 1.5.0
        </div>
    </div>
    {sections_html}
</div>
<footer>
    pbg-pennylane-data-reuploading &middot; MIT License &middot;
    <a href="https://github.com/vivarium-collective/pbg-pennylane-data-reuploading">GitHub</a>
</footer>
</body>
</html>"""

    with open(REPORT_PATH, "w") as f:
        f.write(html)

    print(f"\nReport written to {REPORT_PATH}")
    webbrowser.open("file://" + os.path.abspath(REPORT_PATH))


if __name__ == "__main__":
    generate_report()
