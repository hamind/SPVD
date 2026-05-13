from __future__ import annotations

import numpy as np
import pandas as pd


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _std(series: pd.Series) -> float:
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1))


def summarize_aro(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "category", "accuracy", "mean_margin", "median_margin", "margin_std", "negative_win_rate", "sample_count"]
    df = pairwise_df[pairwise_df["benchmark"] == "aro"] if not pairwise_df.empty else pd.DataFrame()
    if df.empty:
        return _empty(columns)
    rows = []
    for (model_name, category), group in df.groupby(["model_name", "category"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "category": category,
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    overall = []
    for model_name, group in df.groupby("model_name", dropna=False):
        overall.append(
            {
                "model_name": model_name,
                "category": "overall",
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows + overall, columns=columns)


def summarize_sugarcrepe(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "category", "subcategory", "accuracy", "mean_margin", "median_margin", "margin_std", "negative_win_rate", "sample_count"]
    df = pairwise_df[pairwise_df["benchmark"] == "sugarcrepe"] if not pairwise_df.empty else pd.DataFrame()
    if df.empty:
        return _empty(columns)
    rows = []
    for (model_name, category, subcategory), group in df.groupby(["model_name", "category", "subcategory"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "category": category,
                "subcategory": subcategory,
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    for model_name, group in df.groupby("model_name", dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "category": "overall",
                "subcategory": np.nan,
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_pairwise_by_benchmark(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "benchmark", "category", "subcategory", "accuracy", "mean_margin", "median_margin", "margin_std", "negative_win_rate", "sample_count"]
    if pairwise_df.empty:
        return _empty(columns)
    rows = []
    for (model_name, benchmark, category, subcategory), group in pairwise_df.groupby(["model_name", "benchmark", "category", "subcategory"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": category,
                "subcategory": subcategory,
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    for (model_name, benchmark), group in pairwise_df.groupby(["model_name", "benchmark"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": "overall",
                "subcategory": np.nan,
                "accuracy": float(group["correct"].mean()),
                "mean_margin": float(group["hard_margin"].mean()),
                "median_margin": float(group["hard_margin"].median()),
                "margin_std": _std(group["hard_margin"]),
                "negative_win_rate": float((group["score_neg"] >= group["score_pos"]).mean()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_winoground(winoground_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "text_score", "image_score", "group_score", "group_min_margin_mean", "group_min_margin_median", "sample_count"]
    if winoground_df.empty:
        return _empty(columns)
    if "benchmark" in winoground_df.columns:
        winoground_df = winoground_df[winoground_df["benchmark"] == "winoground"]
    if winoground_df.empty:
        return _empty(columns)
    rows = []
    for model_name, group in winoground_df.groupby("model_name", dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "text_score": float(group["text_score"].mean()),
                "image_score": float(group["image_score"].mean()),
                "group_score": float(group["group_score"].mean()),
                "group_min_margin_mean": float(group["group_min_margin"].mean()),
                "group_min_margin_median": float(group["group_min_margin"].median()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_2x2_by_benchmark(winoground_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "benchmark", "category", "text_score", "image_score", "group_score", "group_min_margin_mean", "group_min_margin_median", "sample_count"]
    if winoground_df.empty:
        return _empty(columns)
    rows = []
    for (model_name, benchmark, category), group in winoground_df.groupby(["model_name", "benchmark", "category"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": category,
                "text_score": float(group["text_score"].mean()),
                "image_score": float(group["image_score"].mean()),
                "group_score": float(group["group_score"].mean()),
                "group_min_margin_mean": float(group["group_min_margin"].mean()),
                "group_min_margin_median": float(group["group_min_margin"].median()),
                "sample_count": int(len(group)),
            }
        )
    for (model_name, benchmark), group in winoground_df.groupby(["model_name", "benchmark"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": "overall",
                "text_score": float(group["text_score"].mean()),
                "image_score": float(group["image_score"].mean()),
                "group_score": float(group["group_score"].mean()),
                "group_min_margin_mean": float(group["group_min_margin"].mean()),
                "group_min_margin_median": float(group["group_min_margin"].median()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_margins(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "benchmark",
        "category",
        "hard_margin_mean",
        "hard_margin_median",
        "random_margin_mean",
        "random_margin_median",
        "hard_vs_random_gap",
        "sample_count",
    ]
    if pairwise_df.empty:
        return _empty(columns)
    df = pairwise_df
    if df.empty:
        return _empty(columns)
    rows = []
    for (model_name, benchmark, category), group in df.groupby(["model_name", "benchmark", "category"], dropna=False):
        hard_mean = float(group["hard_margin"].mean())
        random_mean = float(group["random_margin"].mean()) if "random_margin" in group else float("nan")
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": category,
                "hard_margin_mean": hard_mean,
                "hard_margin_median": float(group["hard_margin"].median()),
                "random_margin_mean": random_mean,
                "random_margin_median": float(group["random_margin"].median()) if "random_margin" in group else float("nan"),
                "hard_vs_random_gap": random_mean - hard_mean if np.isfinite(random_mean) else float("nan"),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_ssr(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "benchmark", "category", "SSR_mean", "SSR_median", "invalid_random_margin_ratio", "sample_count"]
    if pairwise_df.empty or "ssr" not in pairwise_df:
        return _empty(columns)
    df = pairwise_df
    if df.empty:
        return _empty(columns)
    rows = []
    for (model_name, benchmark, category), group in df.groupby(["model_name", "benchmark", "category"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "benchmark": benchmark,
                "category": category,
                "SSR_mean": float(group["ssr"].mean()),
                "SSR_median": float(group["ssr"].median()),
                "invalid_random_margin_ratio": float((group["random_margin"] <= 0).mean()),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summary_all_models(
    model_info: dict[str, dict],
    aro_summary: pd.DataFrame,
    sugar_summary: pd.DataFrame,
    winoground_summary: pd.DataFrame,
    margins_summary: pd.DataFrame,
    ssr_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame | None = None,
    two_by_two_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "model_type",
        "ARO_overall_acc",
        "ARO_attr_acc",
        "ARO_rel_acc",
        "ARO_order_acc",
        "ARO_mean_margin",
        "SugarCrepe_overall_acc",
        "SugarCrepe_mean_margin",
        "SugarCrepePP_overall_acc",
        "SugarCrepePP_mean_margin",
        "Winoground_text_score",
        "Winoground_image_score",
        "Winoground_group_score",
        "Winoground_group_min_margin",
        "BiVLC_i2t_score",
        "BiVLC_t2i_score",
        "BiVLC_group_score",
        "BiVLC_group_min_margin",
        "Hard_margin_mean",
        "Random_margin_mean",
        "SSR_mean",
        "SSR_median",
    ]
    rows = []
    names = sorted(model_info)
    for model_name in names:
        row = {"model_name": model_name, "model_type": model_info.get(model_name, {}).get("model_type")}
        model_aro = aro_summary[aro_summary["model_name"] == model_name] if not aro_summary.empty else pd.DataFrame()
        for category, col in [("overall", "ARO_overall_acc"), ("attr", "ARO_attr_acc"), ("rel", "ARO_rel_acc"), ("order", "ARO_order_acc")]:
            match = model_aro[model_aro["category"] == category] if not model_aro.empty else pd.DataFrame()
            row[col] = float(match["accuracy"].iloc[0]) if not match.empty else np.nan
        match = model_aro[model_aro["category"] == "overall"] if not model_aro.empty else pd.DataFrame()
        row["ARO_mean_margin"] = float(match["mean_margin"].iloc[0]) if not match.empty else np.nan
        model_sugar = sugar_summary[(sugar_summary["model_name"] == model_name) & (sugar_summary["category"] == "overall")] if not sugar_summary.empty else pd.DataFrame()
        row["SugarCrepe_overall_acc"] = float(model_sugar["accuracy"].iloc[0]) if not model_sugar.empty else np.nan
        row["SugarCrepe_mean_margin"] = float(model_sugar["mean_margin"].iloc[0]) if not model_sugar.empty else np.nan
        if pairwise_summary is not None and not pairwise_summary.empty:
            model_sugar_pp = pairwise_summary[
                (pairwise_summary["model_name"] == model_name)
                & (pairwise_summary["benchmark"] == "sugarcrepe_pp")
                & (pairwise_summary["category"] == "overall")
            ]
        else:
            model_sugar_pp = pd.DataFrame()
        row["SugarCrepePP_overall_acc"] = float(model_sugar_pp["accuracy"].iloc[0]) if not model_sugar_pp.empty else np.nan
        row["SugarCrepePP_mean_margin"] = float(model_sugar_pp["mean_margin"].iloc[0]) if not model_sugar_pp.empty else np.nan
        model_wino = winoground_summary[winoground_summary["model_name"] == model_name] if not winoground_summary.empty else pd.DataFrame()
        row["Winoground_text_score"] = float(model_wino["text_score"].iloc[0]) if not model_wino.empty else np.nan
        row["Winoground_image_score"] = float(model_wino["image_score"].iloc[0]) if not model_wino.empty else np.nan
        row["Winoground_group_score"] = float(model_wino["group_score"].iloc[0]) if not model_wino.empty else np.nan
        row["Winoground_group_min_margin"] = float(model_wino["group_min_margin_mean"].iloc[0]) if not model_wino.empty else np.nan
        if two_by_two_summary is not None and not two_by_two_summary.empty:
            model_bivlc = two_by_two_summary[
                (two_by_two_summary["model_name"] == model_name)
                & (two_by_two_summary["benchmark"] == "bivlc")
                & (two_by_two_summary["category"] == "overall")
            ]
        else:
            model_bivlc = pd.DataFrame()
        row["BiVLC_i2t_score"] = float(model_bivlc["text_score"].iloc[0]) if not model_bivlc.empty else np.nan
        row["BiVLC_t2i_score"] = float(model_bivlc["image_score"].iloc[0]) if not model_bivlc.empty else np.nan
        row["BiVLC_group_score"] = float(model_bivlc["group_score"].iloc[0]) if not model_bivlc.empty else np.nan
        row["BiVLC_group_min_margin"] = float(model_bivlc["group_min_margin_mean"].iloc[0]) if not model_bivlc.empty else np.nan
        model_margins = margins_summary[margins_summary["model_name"] == model_name] if not margins_summary.empty else pd.DataFrame()
        row["Hard_margin_mean"] = float(model_margins["hard_margin_mean"].mean()) if not model_margins.empty else np.nan
        row["Random_margin_mean"] = float(model_margins["random_margin_mean"].mean()) if not model_margins.empty else np.nan
        model_ssr = ssr_summary[ssr_summary["model_name"] == model_name] if not ssr_summary.empty else pd.DataFrame()
        row["SSR_mean"] = float(model_ssr["SSR_mean"].mean()) if not model_ssr.empty else np.nan
        row["SSR_median"] = float(model_ssr["SSR_median"].median()) if not model_ssr.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
