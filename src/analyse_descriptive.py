#!/usr/bin/env python3
"""Run descriptive statistics and visualisations for the phonetics project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist
from scipy.stats import chi2, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
SELECTED_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "y", "u", "o", "ø", "œ"]


def _group_label(df: pd.DataFrame) -> pd.Series:
    return df["l1_status"].astype(str) + "/" + df["gender"].astype(str).str.upper()


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def acoustic_descriptives(acoustic: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    vowels = acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)].copy()
    vowels["speaker_group"] = _group_label(vowels)
    vowels["phoneme_label"] = pd.Categorical(vowels["phoneme_label"], categories=ORAL_VOWELS, ordered=True)

    stats = (
        vowels.groupby(["phoneme_label", "speaker_group"], observed=True)
        .agg(
            n_tokens=("phoneme_label", "size"),
            f1_mean=("f1_lobanov", "mean"),
            f1_median=("f1_lobanov", "median"),
            f1_sd=("f1_lobanov", "std"),
            f1_iqr=("f1_lobanov", lambda values: values.quantile(0.75) - values.quantile(0.25)),
            f1_cv=("f1_lobanov", lambda values: values.std() / values.mean() if values.mean() else np.nan),
            f2_mean=("f2_lobanov", "mean"),
            f2_median=("f2_lobanov", "median"),
            f2_sd=("f2_lobanov", "std"),
            f2_iqr=("f2_lobanov", lambda values: values.quantile(0.75) - values.quantile(0.25)),
            f2_cv=("f2_lobanov", lambda values: values.std() / values.mean() if values.mean() else np.nan),
        )
        .reset_index()
    )
    stats.to_csv(tables_dir / "acoustic_vowel_descriptives.csv", index=False)

    variance = []
    for phoneme, subset in vowels.groupby("phoneme_label", observed=True):
        total = subset["f1_lobanov"].var(ddof=1)
        speaker_means = subset.groupby("speaker_id")["f1_lobanov"].mean()
        inter = speaker_means.var(ddof=1)
        intra = subset.groupby("speaker_id")["f1_lobanov"].var(ddof=1).mean()
        residual = total - inter - intra
        variance.append(
            {
                "phoneme_label": phoneme,
                "total_variance_f1": total,
                "inter_speaker_variance_f1": inter,
                "intra_speaker_variance_f1": intra,
                "residual_variance_f1": residual,
            }
        )
    pd.DataFrame(variance).to_csv(tables_dir / "acoustic_f1_variance_decomposition.csv", index=False)

    _plot_vowel_chart(vowels, figures_dir / "vowel_chart_lobanov.png")
    _plot_boxplots(vowels, figures_dir)
    _plot_intra_speaker_variability(vowels, figures_dir / "intra_speaker_variability_violin.png")


def _plot_vowel_chart(vowels: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    palette = dict(zip(sorted(vowels["speaker_group"].unique()), sns.color_palette("Set2", 4)))

    for group, subset in vowels.groupby("speaker_group"):
        centroids = subset.groupby("phoneme_label", observed=True)[["f2_lobanov", "f1_lobanov"]].mean()
        ax.scatter(centroids["f2_lobanov"], centroids["f1_lobanov"], s=55, label=group, color=palette[group])
        for phoneme, row in centroids.iterrows():
            ax.text(row["f2_lobanov"], row["f1_lobanov"], str(phoneme), fontsize=10)

        coords = subset[["f2_lobanov", "f1_lobanov"]].dropna().to_numpy()
        if coords.shape[0] > 2:
            _add_confidence_ellipse(ax, coords, palette[group])

    ax.invert_yaxis()
    ax.set_xlabel("F2 Lobanov")
    ax.set_ylabel("F1 Lobanov")
    ax.set_title("French oral vowel centroids by speaker group")
    ax.legend(title="Group", frameon=False)
    _savefig(output)


def _add_confidence_ellipse(ax: plt.Axes, coords: np.ndarray, color: tuple[float, float, float]) -> None:
    cov = np.cov(coords, rowvar=False)
    if not np.isfinite(cov).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(0.95, df=2))
    width, height = 2 * scale * np.sqrt(np.maximum(eigenvalues, 0))
    ellipse = Ellipse(
        xy=coords.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
        alpha=0.8,
    )
    ax.add_patch(ellipse)


def _plot_boxplots(vowels: pd.DataFrame, figures_dir: Path) -> None:
    for formant in ("f1_lobanov", "f2_lobanov"):
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=vowels, x="phoneme_label", y=formant, hue="speaker_group", fliersize=1)
        plt.title(f"{formant} by oral vowel and speaker group")
        plt.xlabel("Phoneme")
        plt.ylabel(formant)
        plt.legend(title="Group", frameon=False, ncol=4)
        _savefig(figures_dir / f"{formant}_boxplot_by_group.png")


def _plot_intra_speaker_variability(vowels: pd.DataFrame, output: Path) -> None:
    selected = vowels[vowels["phoneme_label"].isin(SELECTED_VOWELS)]
    variability = (
        selected.groupby(["speaker_id", "speaker_group", "phoneme_label"], observed=True)
        .agg(f1_sd=("f1_lobanov", "std"), f2_sd=("f2_lobanov", "std"))
        .reset_index()
    )
    variability["mean_formant_sd"] = variability[["f1_sd", "f2_sd"]].mean(axis=1)
    plt.figure(figsize=(12, 5))
    sns.violinplot(data=variability, x="phoneme_label", y="mean_formant_sd", hue="speaker_group", cut=0)
    plt.title("Intra-speaker variability across repetitions")
    plt.xlabel("Phoneme")
    plt.ylabel("Mean SD of F1/F2 Lobanov")
    plt.legend(title="Group", frameon=False, ncol=4)
    _savefig(output)


def neural_descriptives(
    tokens: pd.DataFrame,
    whisper_pca: Path,
    xlsr_pca: Path,
    tables_dir: Path,
    figures_dir: Path,
    random_state: int,
) -> None:
    token_meta = tokens[["phoneme_label", "l1_status", "gender"]].copy()
    token_meta["speaker_group"] = _group_label(token_meta)

    rows = []
    for model_name, pca_path in [("whisper", whisper_pca), ("xlsr", xlsr_pca)]:
        with np.load(pca_path) as data:
            for key in sorted(k for k in data.files if k.startswith("pca2_layer_")):
                layer = key.removeprefix("pca2_layer_")
                coords = data[key]
                rows.append(_projection_metrics(model_name, layer, "pca", coords, token_meta))
                _plot_projection_grid(
                    coords,
                    token_meta,
                    figures_dir / f"{model_name}_layer_{layer}_pca2.png",
                    title=f"{model_name.upper()} layer {layer} PCA",
                )

                pca50_key = f"pca50_layer_{layer}"
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=30,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=random_state,
                )
                umap_coords = reducer.fit_transform(data[pca50_key]).astype(np.float32)
                rows.append(_projection_metrics(model_name, layer, "umap", umap_coords, token_meta))
                _plot_projection_grid(
                    umap_coords,
                    token_meta,
                    figures_dir / f"{model_name}_layer_{layer}_umap2.png",
                    title=f"{model_name.upper()} layer {layer} UMAP",
                )

    pd.DataFrame(rows).to_csv(tables_dir / "neural_projection_metrics.csv", index=False)
    _mantel_rsm_sample(tokens, whisper_pca, xlsr_pca, tables_dir, random_state)


def _projection_metrics(
    model_name: str,
    layer: str,
    method: str,
    coords: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, float | str]:
    phonemes = meta["phoneme_label"].astype(str).to_numpy()
    grand = coords.mean(axis=0)
    total_ss = float(np.square(coords - grand).sum())
    between_ss = 0.0
    for phoneme in np.unique(phonemes):
        mask = phonemes == phoneme
        center = coords[mask].mean(axis=0)
        between_ss += float(mask.sum() * np.square(center - grand).sum())

    sim = _sample_cosine_similarity(coords, phonemes)
    return {
        "model": model_name,
        "layer": layer,
        "method": method,
        "between_phoneme_variance_ratio_2d": between_ss / total_ss if total_ss else np.nan,
        **sim,
    }


def _sample_cosine_similarity(
    values: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 20_000,
    random_state: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    values = values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    n = len(values)
    same_scores = []
    diff_scores = []
    label_to_indices = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    usable_same = [indices for indices in label_to_indices.values() if len(indices) >= 2]
    unique_labels = np.array(list(label_to_indices.keys()))

    for _ in range(n_pairs):
        indices = usable_same[rng.integers(0, len(usable_same))]
        i, j = rng.choice(indices, size=2, replace=False)
        same_scores.append(float(np.dot(values[i], values[j])))

        label_a, label_b = rng.choice(unique_labels, size=2, replace=False)
        i = rng.choice(label_to_indices[label_a])
        j = rng.choice(label_to_indices[label_b])
        diff_scores.append(float(np.dot(values[i], values[j])))

    same = float(np.mean(same_scores))
    diff = float(np.mean(diff_scores))
    return {
        "within_phoneme_cosine_mean": same,
        "between_phoneme_cosine_mean": diff,
        "within_between_similarity_ratio": same / diff if diff else np.nan,
        "similarity_sample_pairs": n_pairs,
    }


def _plot_projection_grid(coords: np.ndarray, meta: pd.DataFrame, output: Path, title: str) -> None:
    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "phoneme": meta["phoneme_label"].astype(str),
            "l1_status": meta["l1_status"].astype(str),
            "gender": meta["gender"].astype(str),
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, hue, subtitle in zip(axes, ["phoneme", "l1_status", "gender"], ["Phoneme", "L1 Status", "Gender"]):
        sns.scatterplot(data=plot_df, x="x", y="y", hue=hue, s=8, linewidth=0, alpha=0.55, ax=ax, legend=False)
        ax.set_title(subtitle)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    fig.suptitle(title)
    _savefig(output)


def _mantel_rsm_sample(
    tokens: pd.DataFrame,
    whisper_pca: Path,
    xlsr_pca: Path,
    tables_dir: Path,
    random_state: int,
    n_sample: int = 3000,
) -> None:
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(len(tokens), size=min(n_sample, len(tokens)), replace=False))

    acoustic = pd.read_csv("data/features_acoustic_norm.csv", low_memory=False)
    acoustic_values = acoustic.loc[sample_idx, ["f1_lobanov", "f2_lobanov"]].fillna(0).to_numpy()
    acoustic_rsm = -pdist(acoustic_values, metric="euclidean")

    with np.load(whisper_pca) as whisper, np.load(xlsr_pca) as xlsr:
        whisper_values = whisper["pca50_layer_20"][sample_idx]
        xlsr_values = xlsr["pca50_layer_18"][sample_idx]
        whisper_rsm = 1 - pdist(whisper_values, metric="cosine")
        xlsr_rsm = 1 - pdist(xlsr_values, metric="cosine")

    rows = [
        _mantel_row("acoustic", "whisper_layer20", acoustic_rsm, whisper_rsm, len(sample_idx)),
        _mantel_row("acoustic", "xlsr_layer18", acoustic_rsm, xlsr_rsm, len(sample_idx)),
        _mantel_row("whisper_layer20", "xlsr_layer18", whisper_rsm, xlsr_rsm, len(sample_idx)),
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "rsm_mantel_sample.csv", index=False)


def _mantel_row(name_a: str, name_b: str, values_a: np.ndarray, values_b: np.ndarray, n_sample: int) -> dict[str, object]:
    result = spearmanr(values_a, values_b)
    return {
        "representation_a": name_a,
        "representation_b": name_b,
        "spearman_mantel_r": result.statistic,
        "p_value_asymptotic": result.pvalue,
        "n_tokens_sampled": n_sample,
        "n_pairwise_values": len(values_a),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path("data/phoneme_tokens.csv"))
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--whisper-pca", type=Path, default=Path("data/features_whisper_pca.npz"))
    parser.add_argument("--xlsr-pca", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs(args.tables_dir, args.figures_dir)

    acoustic = pd.read_csv(args.acoustic, low_memory=False)
    tokens = pd.read_csv(args.tokens, low_memory=False)

    acoustic_descriptives(acoustic, args.tables_dir, args.figures_dir)
    neural_descriptives(tokens, args.whisper_pca, args.xlsr_pca, args.tables_dir, args.figures_dir, args.random_state)

    manifest = {
        "tables": sorted(path.name for path in args.tables_dir.glob("*.csv")),
        "figures": sorted(path.name for path in args.figures_dir.glob("*.png")),
    }
    (args.tables_dir / "descriptive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
