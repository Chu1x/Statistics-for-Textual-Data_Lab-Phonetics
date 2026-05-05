#!/usr/bin/env python3
"""Hierarchical clustering analyses for phonemes and speakers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
CONSONANTS = ["p", "t", "k", "f", "s", "ʃ", "ʒ", "ʁ", "l", "n"]
PHONEME_SET = ORAL_VOWELS + CONSONANTS
FRONT_BACK_CENTRAL = {
    "i": "front",
    "e": "front",
    "ɛ": "front",
    "a": "front",
    "ɑ": "back",
    "u": "back",
    "o": "back",
    "y": "central_rounded",
    "ø": "central_rounded",
    "œ": "central_rounded",
    "ə": "central",
}
HEIGHT = {
    "i": "high",
    "y": "high",
    "u": "high",
    "e": "mid",
    "ø": "mid",
    "o": "mid",
    "ɛ": "mid",
    "œ": "mid",
    "ə": "mid",
    "a": "low",
    "ɑ": "low",
}


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _load_arrays(whisper_pca: Path, xlsr_pca: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(whisper_pca) as whisper, np.load(xlsr_pca) as xlsr:
        return whisper["pca50_layer_20"], xlsr["pca50_layer_18"]


def _cluster(values: np.ndarray, metric: str, linkage_method: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    distances = pdist(values, metric=metric)
    z = linkage(distances, method=linkage_method)
    labels = fcluster(z, t=k, criterion="maxclust")
    return labels, z


def _best_silhouette(values: np.ndarray, metric: str, linkage_method: str, max_k: int = 8) -> tuple[int, float]:
    best_k = 2
    best_score = np.nan
    for k in range(2, min(max_k, len(values) - 1) + 1):
        labels, _ = _cluster(values, metric, linkage_method, k)
        if len(np.unique(labels)) < 2:
            continue
        distances = squareform(pdist(values, metric=metric))
        score = silhouette_score(distances, labels, metric="precomputed")
        if not np.isfinite(best_score) or score > best_score:
            best_k = k
            best_score = float(score)
    return best_k, best_score


def _plot_dendrogram(z: np.ndarray, labels: list[str], title: str, output: Path) -> None:
    plt.figure(figsize=(9, 5))
    dendrogram(z, labels=labels, leaf_rotation=0)
    plt.title(title)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def phoneme_vowel_clustering(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows = []
    label_rows = []
    reps = _vowel_representations(acoustic, meta, whisper, xlsr)
    for representation, values in reps.items():
        metric, method = ("euclidean", "ward") if representation == "acoustic" else ("cosine", "average")
        phonemes = [p for p in ORAL_VOWELS if p in values.index]
        matrix = values.loc[phonemes].to_numpy()
        best_k, best_score = _best_silhouette(matrix, metric, method)
        labels_3, z = _cluster(matrix, metric, method, k=3)
        labels_height, _ = _cluster(matrix, metric, method, k=3)
        labels_fb, _ = _cluster(matrix, metric, method, k=4)
        truth_height = [HEIGHT[p] for p in phonemes]
        truth_fb = [FRONT_BACK_CENTRAL[p] for p in phonemes]
        rows.append(
            {
                "analysis": "vowels",
                "representation": representation,
                "metric": metric,
                "linkage": method,
                "best_k_silhouette": best_k,
                "best_silhouette": best_score,
                "ari_height_k3": adjusted_rand_score(truth_height, labels_height),
                "ari_front_back_central_k4": adjusted_rand_score(truth_fb, labels_fb),
                "n_phonemes": len(phonemes),
            }
        )
        for phoneme, cluster3, cluster_fb in zip(phonemes, labels_3, labels_fb):
            label_rows.append(
                {
                    "analysis": "vowels",
                    "representation": representation,
                    "phoneme_label": phoneme,
                    "cluster_k3": int(cluster3),
                    "cluster_front_back_k4": int(cluster_fb),
                    "truth_height": HEIGHT[phoneme],
                    "truth_front_back_central": FRONT_BACK_CENTRAL[phoneme],
                }
            )
        _plot_dendrogram(z, phonemes, f"Vowel clustering: {representation}", figures_dir / f"dendrogram_vowels_{representation}.png")

    pd.DataFrame(rows).to_csv(tables_dir / "clustering_vowel_ari.csv", index=False)
    pd.DataFrame(label_rows).to_csv(tables_dir / "clustering_vowel_labels.csv", index=False)


def _vowel_representations(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
) -> dict[str, pd.DataFrame]:
    mask = meta["phoneme_label"].isin(ORAL_VOWELS).to_numpy()
    labels = meta.loc[mask, "phoneme_label"].to_numpy()
    return {
        "acoustic": _centroid_frame(acoustic.loc[mask, ["f1_lobanov", "f2_lobanov"]].to_numpy(), labels),
        "whisper_layer20": _centroid_frame(whisper[mask], labels),
        "xlsr_layer18": _centroid_frame(xlsr[mask], labels),
    }


def _centroid_frame(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    names = []
    for label in sorted(np.unique(labels), key=lambda x: PHONEME_SET.index(x) if x in PHONEME_SET else 999):
        rows.append(np.nanmean(values[labels == label], axis=0))
        names.append(label)
    return pd.DataFrame(np.vstack(rows), index=names)


def consonant_vowel_clustering(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows = []
    label_rows = []
    reps = _phoneme_representations(acoustic, meta, whisper, xlsr)
    for representation, values in reps.items():
        metric, method = ("euclidean", "ward") if representation == "acoustic" else ("cosine", "average")
        phonemes = [p for p in PHONEME_SET if p in values.index]
        matrix = values.loc[phonemes].to_numpy()
        labels, z = _cluster(matrix, metric, method, k=2)
        truth = ["vowel" if p in ORAL_VOWELS else "consonant" for p in phonemes]
        ari = adjusted_rand_score(truth, labels)
        distances = squareform(pdist(matrix, metric=metric))
        silhouette = silhouette_score(distances, labels, metric="precomputed")
        rows.append(
            {
                "analysis": "consonant_vowel",
                "representation": representation,
                "metric": metric,
                "linkage": method,
                "k": 2,
                "ari_consonant_vowel": ari,
                "silhouette": silhouette,
                "n_phonemes": len(phonemes),
            }
        )
        for phoneme, cluster, true_label in zip(phonemes, labels, truth):
            label_rows.append(
                {
                    "analysis": "consonant_vowel",
                    "representation": representation,
                    "phoneme_label": phoneme,
                    "cluster_k2": int(cluster),
                    "truth": true_label,
                }
            )
        _plot_dendrogram(z, phonemes, f"Consonant/vowel clustering: {representation}", figures_dir / f"dendrogram_consonant_vowel_{representation}.png")
    pd.DataFrame(rows).to_csv(tables_dir / "clustering_consonant_vowel_ari.csv", index=False)
    pd.DataFrame(label_rows).to_csv(tables_dir / "clustering_consonant_vowel_labels.csv", index=False)


def _phoneme_representations(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
) -> dict[str, pd.DataFrame]:
    mask = meta["phoneme_label"].isin(PHONEME_SET).to_numpy()
    labels = meta.loc[mask, "phoneme_label"].to_numpy()
    acoustic_values = acoustic.loc[mask, ["f1_lobanov", "f2_lobanov", "duration_ms", "scg_hz"]].copy()
    acoustic_values["scg_hz"] = acoustic_values["scg_hz"].fillna(acoustic_values["scg_hz"].median())
    acoustic_values = acoustic_values.fillna(acoustic_values.median(numeric_only=True))
    acoustic_scaled = StandardScaler().fit_transform(acoustic_values.to_numpy())
    return {
        "acoustic": _centroid_frame(acoustic_scaled, labels),
        "whisper_layer20": _centroid_frame(whisper[mask], labels),
        "xlsr_layer18": _centroid_frame(xlsr[mask], labels),
    }


def speaker_clustering(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows = []
    label_rows = []
    reps = _speaker_representations(acoustic, meta, whisper, xlsr)
    speaker_truth = meta[["speaker_id", "l1_status", "gender"]].drop_duplicates().set_index("speaker_id")
    for representation, values in reps.items():
        metric, method = ("euclidean", "ward") if representation == "acoustic" else ("cosine", "average")
        speakers = values.index.tolist()
        matrix = values.to_numpy()
        best_k, best_score = _best_silhouette(matrix, metric, method, max_k=6)
        labels, z = _cluster(matrix, metric, method, k=2)
        l1_truth = speaker_truth.loc[speakers, "l1_status"].to_numpy()
        gender_truth = speaker_truth.loc[speakers, "gender"].to_numpy()
        rows.append(
            {
                "analysis": "speakers",
                "representation": representation,
                "metric": metric,
                "linkage": method,
                "best_k_silhouette": best_k,
                "best_silhouette": best_score,
                "ari_l1_l2_k2": adjusted_rand_score(l1_truth, labels),
                "ari_gender_k2": adjusted_rand_score(gender_truth, labels),
                "n_speakers": len(speakers),
            }
        )
        for speaker, cluster in zip(speakers, labels):
            label_rows.append(
                {
                    "analysis": "speakers",
                    "representation": representation,
                    "speaker_id": speaker,
                    "cluster_k2": int(cluster),
                    "l1_status": speaker_truth.loc[speaker, "l1_status"],
                    "gender": speaker_truth.loc[speaker, "gender"],
                }
            )
        _plot_dendrogram(z, speakers, f"Speaker clustering: {representation}", figures_dir / f"dendrogram_speakers_{representation}.png")
    pd.DataFrame(rows).to_csv(tables_dir / "clustering_speaker_ari.csv", index=False)
    pd.DataFrame(label_rows).to_csv(tables_dir / "clustering_speaker_labels.csv", index=False)


def _speaker_representations(
    acoustic: pd.DataFrame,
    meta: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
) -> dict[str, pd.DataFrame]:
    reps = {}
    mask = meta["phoneme_label"].isin(ORAL_VOWELS).to_numpy()
    for representation, values in [
        ("acoustic", acoustic[["f1_lobanov", "f2_lobanov"]].to_numpy()),
        ("whisper_layer20", whisper),
        ("xlsr_layer18", xlsr),
    ]:
        rows = []
        for speaker, speaker_df in meta.loc[mask].groupby("speaker_id"):
            vector_parts = []
            for phoneme in ORAL_VOWELS:
                phoneme_idx = speaker_df.index[speaker_df["phoneme_label"].eq(phoneme)].to_numpy()
                if len(phoneme_idx) == 0:
                    vector_parts.append(np.full(values.shape[1], np.nan))
                else:
                    vector_parts.append(np.nanmean(values[phoneme_idx], axis=0))
            rows.append((speaker, np.concatenate(vector_parts)))
        frame = pd.DataFrame({speaker: vector for speaker, vector in rows}).T
        frame = frame.fillna(frame.mean(numeric_only=True))
        frame = pd.DataFrame(StandardScaler().fit_transform(frame), index=frame.index)
        reps[representation] = frame
    return reps


def misclassification_summary(tables_dir: Path) -> None:
    vowel = pd.read_csv(tables_dir / "clustering_vowel_labels.csv")
    cv = pd.read_csv(tables_dir / "clustering_consonant_vowel_labels.csv")
    rows = []
    for phoneme in sorted(vowel["phoneme_label"].unique()):
        subset = vowel[vowel["phoneme_label"].eq(phoneme)]
        height_clusters = subset.groupby("cluster_k3")["representation"].nunique().max()
        fb_clusters = subset.groupby("cluster_front_back_k4")["representation"].nunique().max()
        rows.append(
            {
                "phoneme_label": phoneme,
                "same_height_cluster_across_representations": height_clusters == subset["representation"].nunique(),
                "same_front_back_cluster_across_representations": fb_clusters == subset["representation"].nunique(),
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "clustering_systematic_phoneme_patterns.csv", index=False)

    cv["correct_binary"] = cv.apply(
        lambda row: row["truth"] == _majority_truth_for_cluster(cv, row["representation"], row["cluster_k2"]),
        axis=1,
    )
    cv.groupby(["phoneme_label", "truth"]).agg(
        n_representations=("representation", "nunique"),
        n_binary_misclustered=("correct_binary", lambda x: int((~x).sum())),
    ).reset_index().to_csv(tables_dir / "clustering_consonant_vowel_misclustered.csv", index=False)


def _majority_truth_for_cluster(df: pd.DataFrame, representation: str, cluster: int) -> str:
    subset = df[df["representation"].eq(representation) & df["cluster_k2"].eq(cluster)]
    return subset["truth"].value_counts().idxmax()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--whisper-pca", type=Path, default=Path("data/features_whisper_pca.npz"))
    parser.add_argument("--xlsr-pca", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs(args.tables_dir, args.figures_dir)
    acoustic = pd.read_csv(args.acoustic, low_memory=False).reset_index(drop=True)
    meta = acoustic[["speaker_id", "phoneme_label", "l1_status", "gender"]].copy()
    whisper, xlsr = _load_arrays(args.whisper_pca, args.xlsr_pca)

    phoneme_vowel_clustering(acoustic, meta, whisper, xlsr, args.tables_dir, args.figures_dir)
    consonant_vowel_clustering(acoustic, meta, whisper, xlsr, args.tables_dir, args.figures_dir)
    speaker_clustering(acoustic, meta, whisper, xlsr, args.tables_dir, args.figures_dir)
    misclassification_summary(args.tables_dir)

    outputs = sorted(path.name for path in args.tables_dir.glob("clustering*.csv"))
    print(json.dumps({"clustering_tables": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
