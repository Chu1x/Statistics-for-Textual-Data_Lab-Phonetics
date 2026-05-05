#!/usr/bin/env python3
"""Compare midpoint and long-vowel trajectory acoustic conclusions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import levene, mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
TABLES = Path("results/tables")
FIGURES = Path("results/figures")


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _safe_shapiro(values: np.ndarray) -> float:
    if len(values) > 5000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=5000, replace=False)
    return float(shapiro(values).pvalue)


def _empty_test_row(phoneme: str, feature: str, representation: str, n_l1: int, n_l2: int) -> dict[str, object]:
    return {
        "phoneme_label": phoneme,
        "feature": feature,
        "representation": representation,
        "n_l1": n_l1,
        "n_l2": n_l2,
        "mean_l1": np.nan,
        "mean_l2": np.nan,
        "difference_l2_minus_l1": np.nan,
        "shapiro_p_l1": np.nan,
        "shapiro_p_l2": np.nan,
        "levene_p": np.nan,
        "method": "insufficient_data",
        "statistic": np.nan,
        "p_value": np.nan,
    }


def _group_test(subset: pd.DataFrame, phoneme: str, feature: str, representation: str) -> dict[str, object]:
    l1 = subset.loc[subset["l1_status"] == "L1", feature].dropna().to_numpy()
    l2 = subset.loc[subset["l1_status"] == "L2", feature].dropna().to_numpy()
    if len(l1) < 3 or len(l2) < 3:
        return _empty_test_row(phoneme, feature, representation, len(l1), len(l2))

    p_norm_l1 = _safe_shapiro(l1)
    p_norm_l2 = _safe_shapiro(l2)
    p_levene = float(levene(l1, l2, center="median").pvalue)
    assumptions_hold = p_norm_l1 > 0.05 and p_norm_l2 > 0.05 and p_levene > 0.05
    if assumptions_hold:
        test = ttest_ind(l1, l2, equal_var=True)
        method = "two_sample_t"
    else:
        test = mannwhitneyu(l1, l2, alternative="two-sided")
        method = "mann_whitney_u"

    return {
        "phoneme_label": phoneme,
        "feature": feature,
        "representation": representation,
        "n_l1": len(l1),
        "n_l2": len(l2),
        "mean_l1": float(np.mean(l1)),
        "mean_l2": float(np.mean(l2)),
        "difference_l2_minus_l1": float(np.mean(l2) - np.mean(l1)),
        "shapiro_p_l1": p_norm_l1,
        "shapiro_p_l2": p_norm_l2,
        "levene_p": p_levene,
        "method": method,
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def build_trajectory_features(acoustic: pd.DataFrame) -> pd.DataFrame:
    vowels = acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)].copy()
    required = ["f1_25_lobanov", "f1_75_lobanov", "f2_25_lobanov", "f2_75_lobanov"]
    vowels = vowels.dropna(subset=required)
    vowels["f1_traj_mean_lobanov"] = vowels[["f1_25_lobanov", "f1_75_lobanov"]].mean(axis=1)
    vowels["f2_traj_mean_lobanov"] = vowels[["f2_25_lobanov", "f2_75_lobanov"]].mean(axis=1)
    vowels["f1_slope_75_minus_25_lobanov"] = vowels["f1_75_lobanov"] - vowels["f1_25_lobanov"]
    vowels["f2_slope_75_minus_25_lobanov"] = vowels["f2_75_lobanov"] - vowels["f2_25_lobanov"]
    return vowels


def run_tests(vowels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("f1_lobanov", "midpoint"),
        ("f2_lobanov", "midpoint"),
        ("f1_traj_mean_lobanov", "trajectory_mean"),
        ("f2_traj_mean_lobanov", "trajectory_mean"),
        ("f1_slope_75_minus_25_lobanov", "trajectory_slope"),
        ("f2_slope_75_minus_25_lobanov", "trajectory_slope"),
    ]
    for phoneme in ORAL_VOWELS:
        subset = vowels[vowels["phoneme_label"] == phoneme]
        for feature, representation in specs:
            rows.append(_group_test(subset, phoneme, feature, representation))

    results = pd.DataFrame(rows)
    results["p_fdr_bh"] = np.nan
    for representation, index in results.groupby("representation").groups.items():
        del representation
        p_values = results.loc[index, "p_value"].fillna(1.0)
        results.loc[index, "p_fdr_bh"] = multipletests(p_values, method="fdr_bh")[1]
    results["significant_fdr_0_05"] = results["p_value"].notna() & (results["p_fdr_bh"] < 0.05)
    return results


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phoneme in ORAL_VOWELS:
        for axis in ["f1", "f2"]:
            midpoint_feature = f"{axis}_lobanov"
            trajectory_feature = f"{axis}_traj_mean_lobanov"
            midpoint = results[
                (results["phoneme_label"] == phoneme)
                & (results["feature"] == midpoint_feature)
                & (results["representation"] == "midpoint")
            ].iloc[0]
            trajectory = results[
                (results["phoneme_label"] == phoneme)
                & (results["feature"] == trajectory_feature)
                & (results["representation"] == "trajectory_mean")
            ].iloc[0]
            rows.append(
                {
                    "phoneme_label": phoneme,
                    "axis": axis.upper(),
                    "n_l1": int(midpoint.n_l1),
                    "n_l2": int(midpoint.n_l2),
                    "midpoint_difference_l2_minus_l1": midpoint.difference_l2_minus_l1,
                    "trajectory_difference_l2_minus_l1": trajectory.difference_l2_minus_l1,
                    "absolute_difference_change": abs(
                        trajectory.difference_l2_minus_l1 - midpoint.difference_l2_minus_l1
                    )
                    if pd.notna(midpoint.difference_l2_minus_l1) and pd.notna(trajectory.difference_l2_minus_l1)
                    else np.nan,
                    "midpoint_p_fdr_bh": midpoint.p_fdr_bh,
                    "trajectory_p_fdr_bh": trajectory.p_fdr_bh,
                    "midpoint_significant_fdr_0_05": bool(midpoint.significant_fdr_0_05),
                    "trajectory_significant_fdr_0_05": bool(trajectory.significant_fdr_0_05),
                    "conclusion_changed": bool(midpoint.significant_fdr_0_05)
                    != bool(trajectory.significant_fdr_0_05),
                }
            )
    return pd.DataFrame(rows)


def write_plot(comparison: pd.DataFrame, output: Path) -> None:
    plot_data = comparison.melt(
        id_vars=["phoneme_label", "axis"],
        value_vars=["midpoint_difference_l2_minus_l1", "trajectory_difference_l2_minus_l1"],
        var_name="measurement",
        value_name="difference_l2_minus_l1",
    )
    plot_data["measurement"] = plot_data["measurement"].map(
        {
            "midpoint_difference_l2_minus_l1": "Midpoint",
            "trajectory_difference_l2_minus_l1": "25/75 mean",
        }
    )
    plot_data["label"] = plot_data["phoneme_label"] + " " + plot_data["axis"]

    plt.figure(figsize=(11, 5))
    sns.pointplot(
        data=plot_data,
        x="label",
        y="difference_l2_minus_l1",
        hue="measurement",
        dodge=0.35,
        errorbar=None,
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Vowel and formant axis")
    plt.ylabel("L2 - L1 difference (Lobanov z)")
    plt.title("Midpoint vs 25/75 trajectory mean for long vowels")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs(TABLES, FIGURES)
    acoustic = pd.read_csv(args.acoustic, low_memory=False)
    vowels = build_trajectory_features(acoustic)
    results = run_tests(vowels)
    comparison = comparison_table(results)

    results.to_csv(TABLES / "trajectory_l1_l2_tests.csv", index=False)
    comparison.to_csv(TABLES / "trajectory_midpoint_comparison.csv", index=False)
    write_plot(comparison, FIGURES / "trajectory_midpoint_vs_trajectory.png")

    changed = int(comparison["conclusion_changed"].sum())
    total = len(comparison)
    print(f"Wrote trajectory analysis: {changed}/{total} midpoint-vs-trajectory conclusions changed")


if __name__ == "__main__":
    main()
