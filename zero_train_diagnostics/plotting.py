from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _grouped_bar(ax: plt.Axes, labels: list[str], series: dict[str, list[float]], ylabel: str, title: str) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_aro_accuracy(aro_summary: pd.DataFrame, out_dir: Path) -> None:
    if aro_summary.empty:
        return
    models = sorted(aro_summary["model_name"].dropna().unique())
    categories = ["attr", "rel", "order", "overall"]
    series = {}
    for category in categories:
        values = []
        for model in models:
            match = aro_summary[(aro_summary["model_name"] == model) & (aro_summary["category"] == category)]
            values.append(float(match["accuracy"].iloc[0]) if not match.empty else np.nan)
        series[category] = values
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), 4.8))
    _grouped_bar(ax, models, series, "Accuracy", "ARO Accuracy by Model")
    ax.set_ylim(0, 1)
    _save(fig, out_dir, "aro_accuracy_by_model")


def plot_sugar_accuracy(sugar_summary: pd.DataFrame, out_dir: Path) -> None:
    if sugar_summary.empty:
        return
    df = sugar_summary[sugar_summary["category"] != "overall"]
    if df.empty:
        df = sugar_summary
    models = sorted(df["model_name"].dropna().unique())
    categories = sorted(df["category"].dropna().unique())
    series = {}
    for category in categories:
        values = []
        for model in models:
            match = df[(df["model_name"] == model) & (df["category"] == category)]
            values.append(float(match["accuracy"].iloc[0]) if not match.empty else np.nan)
        series[category] = values
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 5.2))
    _grouped_bar(ax, models, series, "Accuracy", "SugarCrepe Accuracy by Model")
    ax.set_ylim(0, 1)
    _save(fig, out_dir, "sugarcrepe_accuracy_by_model")


def plot_winoground_scores(winoground_summary: pd.DataFrame, out_dir: Path) -> None:
    if winoground_summary.empty:
        return
    models = sorted(winoground_summary["model_name"].dropna().unique())
    series = {}
    for col, label in [("text_score", "text"), ("image_score", "image"), ("group_score", "group")]:
        series[label] = [float(winoground_summary[winoground_summary["model_name"] == model][col].iloc[0]) for model in models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), 4.8))
    _grouped_bar(ax, models, series, "Score", "Winoground Scores by Model")
    ax.set_ylim(0, 1)
    _save(fig, out_dir, "winoground_scores_by_model")


def plot_hard_vs_random(margins_summary: pd.DataFrame, out_dir: Path) -> None:
    if margins_summary.empty:
        return
    grouped = margins_summary.groupby("model_name", dropna=False)[["hard_margin_mean", "random_margin_mean"]].mean().reset_index()
    models = grouped["model_name"].tolist()
    series = {
        "hard_margin": grouped["hard_margin_mean"].astype(float).tolist(),
        "random_margin": grouped["random_margin_mean"].astype(float).tolist(),
    }
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), 4.8))
    _grouped_bar(ax, models, series, "Mean margin", "Hard Negative Margin vs Random Negative Margin")
    _save(fig, out_dir, "hard_vs_random_margin_by_model")


def plot_ssr_by_category(ssr_summary: pd.DataFrame, out_dir: Path) -> None:
    if ssr_summary.empty:
        return
    df = ssr_summary.copy()
    df["label"] = df["benchmark"].astype(str) + ":" + df["category"].astype(str)
    labels = sorted(df["label"].dropna().unique())
    models = sorted(df["model_name"].dropna().unique())
    series = {}
    for model in models:
        values = []
        for label in labels:
            match = df[(df["model_name"] == model) & (df["label"] == label)]
            values.append(float(match["SSR_mean"].iloc[0]) if not match.empty else np.nan)
        series[model] = values
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 5.2))
    _grouped_bar(ax, labels, series, "SSR", "Semantic Sensitivity Ratio by Category")
    _save(fig, out_dir, "ssr_by_category")


def plot_margin_distribution(pairwise_df: pd.DataFrame, benchmark: str, out_dir: Path) -> None:
    df = pairwise_df[pairwise_df["benchmark"] == benchmark] if not pairwise_df.empty else pd.DataFrame()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model, group in df.groupby("model_name", dropna=False):
        margins = group["hard_margin"].dropna().astype(float)
        if margins.empty:
            continue
        ax.hist(margins, bins=30, alpha=0.35, density=True, label=str(model))
    ax.set_title(f"{benchmark.upper()} Hard Margin Distribution")
    ax.set_xlabel("Hard margin")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, out_dir, f"margin_distribution_{benchmark}")


def generate_figures(
    pairwise_df: pd.DataFrame,
    aro_summary: pd.DataFrame,
    sugar_summary: pd.DataFrame,
    winoground_summary: pd.DataFrame,
    margins_summary: pd.DataFrame,
    ssr_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    plot_aro_accuracy(aro_summary, out_dir)
    plot_sugar_accuracy(sugar_summary, out_dir)
    plot_winoground_scores(winoground_summary, out_dir)
    plot_hard_vs_random(margins_summary, out_dir)
    plot_ssr_by_category(ssr_summary, out_dir)
    plot_margin_distribution(pairwise_df, "aro", out_dir)
    plot_margin_distribution(pairwise_df, "sugarcrepe", out_dir)
