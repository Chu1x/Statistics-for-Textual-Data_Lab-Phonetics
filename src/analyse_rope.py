#!/usr/bin/env python3
"""Compute confidence intervals and ROPE classifications."""

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
from scipy.spatial.distance import cdist, pdist
from scipy.stats import t


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def acoustic_contrast_ci(acoustic: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows = []
    for phoneme in ORAL_VOWELS:
        subset = acoustic[acoustic["phoneme_label"].eq(phoneme)]
        for feature in ["f1_hz", "f2_hz"]:
            l1_by_speaker = subset[subset["l1_status"].eq("L1")].groupby("speaker_id")[feature].mean().dropna()
            l2_by_speaker = subset[subset["l1_status"].eq("L2")].groupby("speaker_id")[feature].mean().dropna()
            row = _welch_ci(l1_by_speaker.to_numpy(), l2_by_speaker.to_numpy())
            rows.append(
                {
                    "phoneme_label": phoneme,
                    "feature": feature,
                    "estimate_l2_minus_l1_hz": row["estimate"],
                    "ci95_low_hz": row["ci_low"],
                    "ci95_high_hz": row["ci_high"],
                    "n_l1_speakers": len(l1_by_speaker),
                    "n_l2_speakers": len(l2_by_speaker),
                    "rope_low_hz": -20.0,
                    "rope_high_hz": 20.0,
                    "rope_classification": _classify_rope(row["ci_low"], row["ci_high"], -20.0, 20.0),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(tables_dir / "rope_acoustic_contrasts.csv", index=False)
    _plot_forest(
        results[results["feature"].eq("f1_hz")],
        estimate_col="estimate_l2_minus_l1_hz",
        low_col="ci95_low_hz",
        high_col="ci95_high_hz",
        rope=(-20.0, 20.0),
        output=figures_dir / "forest_acoustic_f1_rope.png",
        title="Acoustic F1 L1/L2 contrasts with ROPE",
        xlabel="L2 - L1 difference (Hz)",
    )
    _plot_forest(
        results[results["feature"].eq("f2_hz")],
        estimate_col="estimate_l2_minus_l1_hz",
        low_col="ci95_low_hz",
        high_col="ci95_high_hz",
        rope=(-20.0, 20.0),
        output=figures_dir / "forest_acoustic_f2_rope.png",
        title="Acoustic F2 L1/L2 contrasts with ROPE",
        xlabel="L2 - L1 difference (Hz)",
    )
    return results


def _welch_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or len(y) < 2:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    estimate = float(np.mean(y) - np.mean(x))
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    se = np.sqrt(vx / len(x) + vy / len(y))
    df_num = (vx / len(x) + vy / len(y)) ** 2
    df_den = (vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1)
    df = df_num / df_den if df_den else np.nan
    critical = t.ppf(0.975, df) if np.isfinite(df) else np.nan
    return {"estimate": estimate, "ci_low": estimate - critical * se, "ci_high": estimate + critical * se}


def neural_contrast_ci(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    representation: str,
    tables_dir: Path,
    figures_dir: Path,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    speakers = meta["speaker_id"].drop_duplicates().to_numpy()
    rows = []
    for phoneme in ORAL_VOWELS:
        phoneme_mask = meta["phoneme_label"].eq(phoneme).to_numpy()
        if phoneme_mask.sum() == 0:
            continue
        observed = _centroid_cosine_distance(embeddings[phoneme_mask], meta.loc[phoneme_mask, "l1_status"].to_numpy())
        noise_floor = _intra_speaker_noise_floor(embeddings[phoneme_mask], meta.loc[phoneme_mask])
        boot = []
        for _ in range(n_bootstrap):
            sampled_speakers = rng.choice(speakers, size=len(speakers), replace=True)
            mask = meta["speaker_id"].isin(sampled_speakers).to_numpy() & phoneme_mask
            value = _centroid_cosine_distance(embeddings[mask], meta.loc[mask, "l1_status"].to_numpy())
            if np.isfinite(value):
                boot.append(value)
        if boot:
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        else:
            ci_low, ci_high = np.nan, np.nan
        rows.append(
            {
                "representation": representation,
                "phoneme_label": phoneme,
                "estimate_l1_l2_cosine_distance": observed,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "rope_low": 0.0,
                "rope_high": noise_floor,
                "rope_classification": _classify_rope(ci_low, ci_high, 0.0, noise_floor),
                "noise_floor_intra_speaker_cosine": noise_floor,
                "n_bootstrap": len(boot),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(tables_dir / f"rope_neural_contrasts_{representation}.csv", index=False)
    _plot_forest(
        results,
        estimate_col="estimate_l1_l2_cosine_distance",
        low_col="ci95_low",
        high_col="ci95_high",
        rope=None,
        output=figures_dir / f"forest_{representation}_rope.png",
        title=f"{representation} L1/L2 centroid distances",
        xlabel="Cosine distance",
    )
    return results


def _centroid_cosine_distance(values: np.ndarray, status: np.ndarray) -> float:
    l1 = values[status == "L1"]
    l2 = values[status == "L2"]
    if len(l1) == 0 or len(l2) == 0:
        return np.nan
    c1 = l1.mean(axis=0)
    c2 = l2.mean(axis=0)
    return float(cdist([c1], [c2], metric="cosine")[0, 0])


def _intra_speaker_noise_floor(values: np.ndarray, meta_subset: pd.DataFrame) -> float:
    distances = []
    for speaker, speaker_rows in meta_subset.reset_index().groupby("speaker_id"):
        idx = speaker_rows.index.to_numpy()
        if len(idx) < 2:
            continue
        speaker_values = values[idx]
        distances.extend(pdist(speaker_values, metric="cosine"))
    if not distances:
        return np.nan
    return float(np.mean(distances))


def _classify_rope(ci_low: float, ci_high: float, rope_low: float, rope_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high) or not np.isfinite(rope_low) or not np.isfinite(rope_high):
        return "Insufficient data"
    if ci_low >= rope_low and ci_high <= rope_high:
        return "Equivalent"
    if ci_high < rope_low or ci_low > rope_high:
        return "Non-equivalent"
    return "Indeterminate"


def _plot_forest(
    df: pd.DataFrame,
    estimate_col: str,
    low_col: str,
    high_col: str,
    rope: tuple[float, float] | None,
    output: Path,
    title: str,
    xlabel: str,
) -> None:
    plot_df = df.copy()
    plot_df["phoneme_label"] = pd.Categorical(plot_df["phoneme_label"], ORAL_VOWELS, ordered=True)
    plot_df = plot_df.sort_values("phoneme_label")

    plt.figure(figsize=(8, 6))
    y = np.arange(len(plot_df))
    colors = {
        "Equivalent": "#2ca25f",
        "Non-equivalent": "#de2d26",
        "Indeterminate": "#756bb1",
        "Insufficient data": "#8c8c8c",
    }
    for i, row in enumerate(plot_df.itertuples(index=False)):
        estimate = getattr(row, estimate_col)
        low = getattr(row, low_col)
        high = getattr(row, high_col)
        classification = getattr(row, "rope_classification")
        if np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high):
            plt.errorbar(
                estimate,
                i,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=colors.get(classification, "black"),
                capsize=3,
            )
    if rope is not None:
        plt.axvspan(rope[0], rope[1], color="#d9f0a3", alpha=0.4, label="ROPE")
    plt.axvline(0, color="black", linewidth=1, alpha=0.5)
    plt.yticks(y, plot_df["phoneme_label"])
    plt.xlabel(xlabel)
    plt.ylabel("Phoneme")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def merge_rope_summary(
    acoustic: pd.DataFrame,
    whisper: pd.DataFrame,
    xlsr: pd.DataFrame,
    significance_path: Path,
    tables_dir: Path,
) -> None:
    acoustic_f1 = acoustic[acoustic["feature"].eq("f1_hz")].copy()
    acoustic_f1["representation"] = "acoustic_f1"
    acoustic_f1 = acoustic_f1.rename(
        columns={
            "estimate_l2_minus_l1_hz": "estimate",
            "ci95_low_hz": "ci95_low",
            "ci95_high_hz": "ci95_high",
            "rope_low_hz": "rope_low",
            "rope_high_hz": "rope_high",
        }
    )
    neural = pd.concat(
        [
            whisper.rename(columns={"estimate_l1_l2_cosine_distance": "estimate"}),
            xlsr.rename(columns={"estimate_l1_l2_cosine_distance": "estimate"}),
        ],
        ignore_index=True,
    )
    summary = pd.concat(
        [
            acoustic_f1[["representation", "phoneme_label", "estimate", "ci95_low", "ci95_high", "rope_low", "rope_high", "rope_classification"]],
            neural[["representation", "phoneme_label", "estimate", "ci95_low", "ci95_high", "rope_low", "rope_high", "rope_classification"]],
        ],
        ignore_index=True,
    )

    if significance_path.exists():
        sig = pd.read_csv(significance_path)
        sig = sig[sig["feature"].eq("f1_lobanov")][["phoneme_label", "p_value", "p_fdr_bh"]]
        summary = summary.merge(sig, on="phoneme_label", how="left")
    summary.to_csv(tables_dir / "rope_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--whisper-pca", type=Path, default=Path("data/features_whisper_pca.npz"))
    parser.add_argument("--xlsr-pca", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs(args.tables_dir, args.figures_dir)

    acoustic = pd.read_csv(args.acoustic, low_memory=False).reset_index(drop=True)
    meta = acoustic[["speaker_id", "phoneme_label", "l1_status"]].copy()
    acoustic_results = acoustic_contrast_ci(acoustic, args.tables_dir, args.figures_dir)

    with np.load(args.whisper_pca) as whisper_data, np.load(args.xlsr_pca) as xlsr_data:
        whisper = whisper_data["pca50_layer_20"]
        xlsr = xlsr_data["pca50_layer_18"]

    whisper_results = neural_contrast_ci(
        meta, whisper, "whisper_layer20", args.tables_dir, args.figures_dir, args.bootstrap, args.random_state
    )
    xlsr_results = neural_contrast_ci(
        meta, xlsr, "xlsr_layer18", args.tables_dir, args.figures_dir, args.bootstrap, args.random_state
    )
    merge_rope_summary(
        acoustic_results,
        whisper_results,
        xlsr_results,
        args.tables_dir / "acoustic_l1_l2_tests.csv",
        args.tables_dir,
    )

    print(
        {
            "acoustic_rows": len(acoustic_results),
            "whisper_rows": len(whisper_results),
            "xlsr_rows": len(xlsr_results),
            "bootstrap": args.bootstrap,
        }
    )


if __name__ == "__main__":
    main()
