from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import EvaluationResult


def _scatter(ax, data: pd.DataFrame, x: str, y: str, title: str) -> None:
    clean = data[[x, y, "factor"]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(clean[x], clean[y], s=20, alpha=0.75, color="#1f77b4")
    if len(clean) >= 3:
        m, b = np.polyfit(clean[x], clean[y], 1)
        xs = np.linspace(clean[x].min(), clean[x].max(), 100)
        ax.plot(xs, m * xs + b, color="#d62728", linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.25)


def write_reports(result: EvaluationResult, factor_sources: pd.DataFrame, output_dir: str | Path, metadata: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result.factor_metrics.merge(factor_sources, on="factor", how="left")
    metrics = metrics.sort_values(["abs_mean_ic", "chosen_ann"], ascending=False)
    metrics.to_csv(out / "factor_metrics.csv", index=False)
    result.daily_ic.to_csv(out / "daily_ic.csv")
    result.correlation_summary.to_csv(out / "correlation_summary.csv", index=False)
    metrics.head(30).to_csv(out / "top30_by_abs_ic.csv", index=False)
    metrics.sort_values("chosen_ann", ascending=False).head(30).to_csv(out / "top30_by_chosen_ann.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    _scatter(axes[0, 0], metrics, "mean_ic", "spread_ann", "mean_ic vs high-minus-low ann")
    _scatter(axes[0, 1], metrics, "abs_mean_ic", "chosen_ann", "abs(mean_ic) vs IC-direction long ann")
    _scatter(axes[1, 0], metrics, "abs_ic_ir", "chosen_ann", "abs(ICIR) vs IC-direction long ann")
    _scatter(axes[1, 1], metrics, "abs_mean_ic", "abs_spread_ann", "abs(mean_ic) vs abs spread ann")
    fig.tight_layout()
    fig.savefig(out / "ic_vs_strategy_scatter.png", bbox_inches="tight")
    plt.close(fig)

    top = metrics.head(20).copy()
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    labels = top["factor"].iloc[::-1]
    ax.barh(labels, top["abs_mean_ic"].iloc[::-1], color="#0B5CAD")
    ax.set_title("Top 20 factors by abs(mean IC)")
    ax.set_xlabel("abs(mean IC)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "top_abs_ic.png", bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        "# Factor IC Study",
        "",
        f"- Factor count: {metadata.get('factor_count')}",
        f"- Row count: {metadata.get('row_count')}",
        f"- Test start: {metadata.get('test_start_date')}",
        f"- End date: {metadata.get('end_date')}",
        "",
        "Key files:",
        "",
        "- `factor_metrics.csv`: IC and corresponding factor strategy metrics.",
        "- `daily_ic.csv`: daily cross-sectional Spearman IC by factor.",
        "- `correlation_summary.csv`: cross-factor correlation tests between IC metrics and strategy returns.",
        "- `ic_vs_strategy_scatter.png`: scatter plots for the main relationships.",
    ]
    (out / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

