#!/usr/bin/env python3
"""Summarise acoustic quality flags for transparent reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
ROUGH_RANGES = {
    "f1_hz": (100.0, 1200.0),
    "f2_hz": (300.0, 4000.0),
    "f0_mean_hz": (50.0, 500.0),
}


def _flag_rows(df: pd.DataFrame, subset_name: str, feature: str, low: float, high: float) -> dict[str, object]:
    values = pd.to_numeric(df[feature], errors="coerce")
    nonmissing = values.notna()
    flagged = nonmissing & ((values < low) | (values > high))
    return {
        "subset": subset_name,
        "feature": feature,
        "rough_low": low,
        "rough_high": high,
        "n_tokens": len(df),
        "n_nonmissing": int(nonmissing.sum()),
        "n_flagged": int(flagged.sum()),
        "flagged_prop_nonmissing": float(flagged.sum() / nonmissing.sum()) if nonmissing.sum() else pd.NA,
        "min_value": float(values[nonmissing].min()) if nonmissing.any() else pd.NA,
        "max_value": float(values[nonmissing].max()) if nonmissing.any() else pd.NA,
    }


def acoustic_quality(acoustic: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all_tokens": acoustic,
        "oral_vowels": acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)],
        "non_oral_tokens": acoustic[~acoustic["phoneme_label"].isin(ORAL_VOWELS)],
    }
    rows = []
    for subset_name, subset in subsets.items():
        for feature, (low, high) in ROUGH_RANGES.items():
            rows.append(_flag_rows(subset, subset_name, feature, low, high))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/acoustic_quality_flags.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acoustic = pd.read_csv(args.acoustic, low_memory=False)
    results = acoustic_quality(acoustic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    oral = results[results["subset"].eq("oral_vowels")]
    flagged = int(oral["n_flagged"].sum())
    print(f"Wrote {args.output}; oral-vowel rough-range flags: {flagged}")


if __name__ == "__main__":
    main()
