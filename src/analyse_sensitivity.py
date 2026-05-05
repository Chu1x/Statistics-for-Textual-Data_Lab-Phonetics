#!/usr/bin/env python3
"""Sensitivity checks for acoustic L1/L2 tests after rough-range filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import levene, mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
ROUGH_RANGES = {
    "f1_lobanov": ("f1_hz", 100.0, 1200.0),
    "f2_lobanov": ("f2_hz", 300.0, 4000.0),
}


def _safe_shapiro(values: np.ndarray) -> float:
    if len(values) > 5000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=5000, replace=False)
    return float(shapiro(values).pvalue)


def _empty_test_row(phoneme: str, feature: str, n_l1: int, n_l2: int, n_excluded: int) -> dict[str, object]:
    return {
        "phoneme_label": phoneme,
        "feature": feature,
        "n_l1": n_l1,
        "n_l2": n_l2,
        "n_excluded_by_range_filter": n_excluded,
        "mean_l1": np.nan,
        "mean_l2": np.nan,
        "difference_l2_minus_l1": np.nan,
        "method": "insufficient_data",
        "statistic": np.nan,
        "p_value": np.nan,
    }


def _test_row(subset: pd.DataFrame, phoneme: str, feature: str, n_excluded: int) -> dict[str, object]:
    l1 = subset.loc[subset["l1_status"] == "L1", feature].dropna().to_numpy()
    l2 = subset.loc[subset["l1_status"] == "L2", feature].dropna().to_numpy()
    if len(l1) < 3 or len(l2) < 3:
        return _empty_test_row(phoneme, feature, len(l1), len(l2), n_excluded)

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
        "n_l1": len(l1),
        "n_l2": len(l2),
        "n_excluded_by_range_filter": n_excluded,
        "mean_l1": float(np.mean(l1)),
        "mean_l2": float(np.mean(l2)),
        "difference_l2_minus_l1": float(np.mean(l2) - np.mean(l1)),
        "method": method,
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def filtered_acoustic_tests(acoustic: pd.DataFrame) -> pd.DataFrame:
    vowels = acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)].copy()
    rows = []
    for phoneme in ORAL_VOWELS:
        phoneme_subset = vowels[vowels["phoneme_label"] == phoneme]
        for feature, (raw_feature, low, high) in ROUGH_RANGES.items():
            values = pd.to_numeric(phoneme_subset[raw_feature], errors="coerce")
            keep = values.notna() & values.between(low, high, inclusive="both")
            filtered = phoneme_subset[keep]
            rows.append(_test_row(filtered, phoneme, feature, int((~keep).sum())))

    results = pd.DataFrame(rows)
    results["p_fdr_bh"] = multipletests(results["p_value"].fillna(1.0), method="fdr_bh")[1]
    results["significant_fdr_0_05"] = results["p_value"].notna() & (results["p_fdr_bh"] < 0.05)
    return results


def compare_to_main(main: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    main_cols = [
        "phoneme_label",
        "feature",
        "n_l1",
        "n_l2",
        "difference_l2_minus_l1",
        "p_fdr_bh",
        "significant_fdr_0_05",
    ]
    filtered_cols = main_cols + ["n_excluded_by_range_filter"]
    comparison = main[main_cols].merge(
        filtered[filtered_cols],
        on=["phoneme_label", "feature"],
        suffixes=("_main", "_range_filtered"),
        how="outer",
    )
    comparison["conclusion_changed"] = (
        comparison["significant_fdr_0_05_main"].astype(bool)
        != comparison["significant_fdr_0_05_range_filtered"].astype(bool)
    )
    comparison["absolute_effect_change"] = (
        comparison["difference_l2_minus_l1_range_filtered"] - comparison["difference_l2_minus_l1_main"]
    ).abs()
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--main-tests", type=Path, default=Path("results/tables/acoustic_l1_l2_tests.csv"))
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("results/tables/acoustic_l1_l2_tests_range_filtered.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("results/tables/acoustic_l1_l2_sensitivity_range_filter.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acoustic = pd.read_csv(args.acoustic, low_memory=False)
    main_tests = pd.read_csv(args.main_tests)
    filtered = filtered_acoustic_tests(acoustic)
    comparison = compare_to_main(main_tests, filtered)
    args.filtered_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.filtered_output, index=False)
    comparison.to_csv(args.comparison_output, index=False)
    print(
        "Wrote sensitivity analysis: "
        f"{int(comparison['conclusion_changed'].sum())}/{len(comparison)} conclusions changed"
    )


if __name__ == "__main__":
    main()
