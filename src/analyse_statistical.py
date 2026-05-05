#!/usr/bin/env python3
"""Run group tests, distance analyses, and phoneme classifiers."""

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
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import levene, mannwhitneyu, shapiro, spearmanr, ttest_ind
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
SELECTED_PAIRS = [("e", "ɛ"), ("y", "u"), ("a", "ɑ"), ("ø", "œ")]


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def acoustic_group_tests(acoustic: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    vowels = acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)].copy()
    for phoneme in ORAL_VOWELS:
        subset = vowels[vowels["phoneme_label"] == phoneme]
        for feature in ["f1_lobanov", "f2_lobanov"]:
            l1 = subset.loc[subset["l1_status"] == "L1", feature].dropna().to_numpy()
            l2 = subset.loc[subset["l1_status"] == "L2", feature].dropna().to_numpy()
            if len(l1) < 3 or len(l2) < 3:
                rows.append(_empty_test_row(phoneme, feature, len(l1), len(l2)))
                continue
            p_norm_l1 = _safe_shapiro(l1)
            p_norm_l2 = _safe_shapiro(l2)
            p_levene = levene(l1, l2, center="median").pvalue
            assumptions_hold = p_norm_l1 > 0.05 and p_norm_l2 > 0.05 and p_levene > 0.05
            if assumptions_hold:
                test = ttest_ind(l1, l2, equal_var=True)
                method = "two_sample_t"
                statistic = test.statistic
                p_value = test.pvalue
            else:
                test = mannwhitneyu(l1, l2, alternative="two-sided")
                method = "mann_whitney_u"
                statistic = test.statistic
                p_value = test.pvalue

            rows.append(
                {
                    "phoneme_label": phoneme,
                    "feature": feature,
                    "n_l1": len(l1),
                    "n_l2": len(l2),
                    "mean_l1": float(np.mean(l1)),
                    "mean_l2": float(np.mean(l2)),
                    "difference_l2_minus_l1": float(np.mean(l2) - np.mean(l1)),
                    "shapiro_p_l1": p_norm_l1,
                    "shapiro_p_l2": p_norm_l2,
                    "levene_p": p_levene,
                    "method": method,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

    results = pd.DataFrame(rows)
    results["p_fdr_bh"] = multipletests(results["p_value"].fillna(1.0), method="fdr_bh")[1]
    results["significant_fdr_0_05"] = results["p_fdr_bh"] < 0.05
    results.to_csv(tables_dir / "acoustic_l1_l2_tests.csv", index=False)
    return results


def gender_residual_tests(acoustic: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    vowels = acoustic[acoustic["phoneme_label"].isin(ORAL_VOWELS)].copy()
    for phoneme in ORAL_VOWELS:
        subset = vowels[vowels["phoneme_label"] == phoneme]
        for feature in ["f1_lobanov", "f2_lobanov"]:
            speaker_means = (
                subset.groupby(["speaker_id", "gender"], dropna=False)[feature]
                .mean()
                .dropna()
                .reset_index()
            )
            female = speaker_means.loc[speaker_means["gender"].astype(str).str.lower().eq("f"), feature].to_numpy()
            male = speaker_means.loc[speaker_means["gender"].astype(str).str.lower().eq("m"), feature].to_numpy()
            if len(female) < 2 or len(male) < 2:
                rows.append(
                    {
                        "phoneme_label": phoneme,
                        "feature": feature,
                        "n_female_speakers": len(female),
                        "n_male_speakers": len(male),
                        "method": "insufficient_data",
                        "p_value": np.nan,
                    }
                )
                continue

            p_norm_f = _safe_shapiro(female) if len(female) >= 3 else 0.0
            p_norm_m = _safe_shapiro(male) if len(male) >= 3 else 0.0
            p_levene = levene(female, male, center="median").pvalue
            assumptions_hold = p_norm_f > 0.05 and p_norm_m > 0.05 and p_levene > 0.05
            if assumptions_hold:
                test = ttest_ind(female, male, equal_var=True)
                method = "speaker_level_two_sample_t"
            else:
                test = mannwhitneyu(female, male, alternative="two-sided")
                method = "speaker_level_mann_whitney_u"

            rows.append(
                {
                    "phoneme_label": phoneme,
                    "feature": feature,
                    "n_female_speakers": len(female),
                    "n_male_speakers": len(male),
                    "mean_female": float(np.mean(female)),
                    "mean_male": float(np.mean(male)),
                    "difference_male_minus_female": float(np.mean(male) - np.mean(female)),
                    "shapiro_p_female": p_norm_f,
                    "shapiro_p_male": p_norm_m,
                    "levene_p": p_levene,
                    "method": method,
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )

    results = pd.DataFrame(rows)
    results["p_fdr_bh"] = multipletests(results["p_value"].fillna(1.0), method="fdr_bh")[1]
    results["significant_fdr_0_05"] = results["p_value"].notna() & (results["p_fdr_bh"] < 0.05)
    results["note"] = "Speaker-level comparison after Lobanov normalisation; gender is not paired in this corpus."
    results.to_csv(tables_dir / "gender_residual_tests.csv", index=False)
    return results


def _empty_test_row(phoneme: str, feature: str, n_l1: int, n_l2: int) -> dict[str, object]:
    return {
        "phoneme_label": phoneme,
        "feature": feature,
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


def _safe_shapiro(values: np.ndarray) -> float:
    if len(values) > 5000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=5000, replace=False)
    return float(shapiro(values).pvalue)


def neural_permutation_tests(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
    tables_dir: Path,
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    speaker_status = meta[["speaker_id", "l1_status"]].drop_duplicates().reset_index(drop=True)
    speakers = speaker_status["speaker_id"].to_numpy()
    labels = speaker_status["l1_status"].to_numpy()

    for phoneme in ORAL_VOWELS:
        mask = meta["phoneme_label"].eq(phoneme).to_numpy()
        subset_meta = meta.loc[mask]
        subset_embeddings = embeddings[mask]
        if subset_embeddings.shape[0] == 0:
            continue

        observed = _centroid_cosine_distance(subset_embeddings, subset_meta["l1_status"].to_numpy())
        if not np.isfinite(observed):
            rows.append(
                {
                    "model": model_name,
                    "phoneme_label": phoneme,
                    "n_tokens": int(mask.sum()),
                    "observed_l1_l2_cosine_distance": observed,
                    "null_mean": np.nan,
                    "null_sd": np.nan,
                    "n_permutations": n_permutations,
                    "p_value": np.nan,
                }
            )
            continue
        null = np.empty(n_permutations, dtype=np.float32)
        speaker_to_pos = {speaker: idx for idx, speaker in enumerate(speakers)}
        speaker_positions = subset_meta["speaker_id"].map(speaker_to_pos).to_numpy()
        for b in range(n_permutations):
            shuffled = rng.permutation(labels)
            permuted_status = shuffled[speaker_positions]
            null[b] = _centroid_cosine_distance(subset_embeddings, permuted_status)
        finite_null = null[np.isfinite(null)]
        p_value = (1.0 + np.sum(finite_null >= observed)) / (len(finite_null) + 1.0)
        rows.append(
            {
                "model": model_name,
                "phoneme_label": phoneme,
                "n_tokens": int(mask.sum()),
                "observed_l1_l2_cosine_distance": observed,
                "null_mean": float(np.mean(null)),
                "null_sd": float(np.std(null, ddof=1)),
                "n_permutations": n_permutations,
                "p_value": p_value,
            }
        )

    results = pd.DataFrame(rows)
    results["p_fdr_bh"] = multipletests(results["p_value"].fillna(1.0), method="fdr_bh")[1]
    results["significant_fdr_0_05"] = results["p_value"].notna() & (results["p_fdr_bh"] < 0.05)
    results.to_csv(tables_dir / f"neural_l1_l2_permutation_{model_name}.csv", index=False)
    return results


def _centroid_cosine_distance(values: np.ndarray, status: np.ndarray) -> float:
    l1 = values[status == "L1"]
    l2 = values[status == "L2"]
    if len(l1) == 0 or len(l2) == 0:
        return np.nan
    c1 = l1.mean(axis=0)
    c2 = l2.mean(axis=0)
    denom = np.linalg.norm(c1) * np.linalg.norm(c2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.dot(c1, c2) / denom)


def distance_matrices(
    meta: pd.DataFrame,
    acoustic: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
) -> None:
    vowel_meta = meta[meta["phoneme_label"].isin(ORAL_VOWELS)].copy()
    idx = vowel_meta.index.to_numpy()
    phonemes = [p for p in ORAL_VOWELS if (vowel_meta["phoneme_label"] == p).any()]

    acoustic_values = acoustic.loc[idx, ["f1_lobanov", "f2_lobanov"]].to_numpy()
    acoustic_centroids = _centroids(acoustic_values, vowel_meta["phoneme_label"].to_numpy(), phonemes)
    whisper_centroids = _centroids(whisper[idx], vowel_meta["phoneme_label"].to_numpy(), phonemes)
    xlsr_centroids = _centroids(xlsr[idx], vowel_meta["phoneme_label"].to_numpy(), phonemes)
    pooled_vi = _pooled_inverse_covariance(acoustic_values, vowel_meta["phoneme_label"].to_numpy(), phonemes)

    matrices = {
        "acoustic_euclidean": squareform(pdist(acoustic_centroids, metric="euclidean")),
        "acoustic_mahalanobis": squareform(pdist(acoustic_centroids, metric="mahalanobis", VI=pooled_vi)),
        "whisper_cosine": squareform(pdist(whisper_centroids, metric="cosine")),
        "xlsr_cosine": squareform(pdist(xlsr_centroids, metric="cosine")),
    }
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=phonemes, columns=phonemes).to_csv(tables_dir / f"distance_matrix_{name}.csv")

    rows = [
        _mantel_row("acoustic_euclidean", "acoustic_mahalanobis", matrices["acoustic_euclidean"], matrices["acoustic_mahalanobis"]),
        _mantel_row("acoustic_euclidean", "whisper_cosine", matrices["acoustic_euclidean"], matrices["whisper_cosine"]),
        _mantel_row("acoustic_euclidean", "xlsr_cosine", matrices["acoustic_euclidean"], matrices["xlsr_cosine"]),
        _mantel_row("acoustic_mahalanobis", "whisper_cosine", matrices["acoustic_mahalanobis"], matrices["whisper_cosine"]),
        _mantel_row("acoustic_mahalanobis", "xlsr_cosine", matrices["acoustic_mahalanobis"], matrices["xlsr_cosine"]),
        _mantel_row("whisper_cosine", "xlsr_cosine", matrices["whisper_cosine"], matrices["xlsr_cosine"]),
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "phoneme_distance_mantel.csv", index=False)
    _bootstrap_distance_pairs(meta, acoustic, whisper, xlsr, tables_dir)


def _centroids(values: np.ndarray, labels: np.ndarray, phonemes: list[str]) -> np.ndarray:
    return np.vstack([values[labels == phoneme].mean(axis=0) for phoneme in phonemes])


def _pooled_inverse_covariance(values: np.ndarray, labels: np.ndarray, phonemes: list[str]) -> np.ndarray:
    covariances = []
    weights = []
    for phoneme in phonemes:
        subset = values[labels == phoneme]
        if len(subset) < 3:
            continue
        covariances.append(np.cov(subset, rowvar=False))
        weights.append(len(subset) - 1)
    if not covariances:
        return np.eye(values.shape[1])
    pooled = np.average(np.stack(covariances), axis=0, weights=np.asarray(weights))
    return np.linalg.pinv(pooled)


def _mantel_row(name_a: str, name_b: str, mat_a: np.ndarray, mat_b: np.ndarray) -> dict[str, object]:
    tri = np.triu_indices_from(mat_a, k=1)
    result = spearmanr(mat_a[tri], mat_b[tri])
    return {
        "distance_a": name_a,
        "distance_b": name_b,
        "mantel_spearman_r": result.statistic,
        "p_value_asymptotic": result.pvalue,
        "n_phoneme_pairs": len(mat_a[tri]),
    }


def _bootstrap_distance_pairs(
    meta: pd.DataFrame,
    acoustic: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
    n_bootstrap: int = 1000,
) -> None:
    rng = np.random.default_rng(42)
    speakers = meta["speaker_id"].drop_duplicates().to_numpy()
    acoustic_values = acoustic[["f1_lobanov", "f2_lobanov"]].to_numpy()
    vowel_mask = meta["phoneme_label"].isin(ORAL_VOWELS).to_numpy()
    acoustic_vi = _pooled_inverse_covariance(
        acoustic_values[vowel_mask],
        meta.loc[vowel_mask, "phoneme_label"].to_numpy(),
        [p for p in ORAL_VOWELS if p in set(meta["phoneme_label"])],
    )
    rows = []
    for p, q in SELECTED_PAIRS:
        if p not in set(meta["phoneme_label"]) or q not in set(meta["phoneme_label"]):
            continue
        for rep_name, values, metric in [
            ("acoustic_euclidean", acoustic_values, "euclidean"),
            ("acoustic_mahalanobis", acoustic_values, "mahalanobis"),
            ("whisper_layer20", whisper, "cosine"),
            ("xlsr_layer18", xlsr, "cosine"),
        ]:
            boot = []
            for _ in range(n_bootstrap):
                sampled_speakers = rng.choice(speakers, size=len(speakers), replace=True)
                mask = meta["speaker_id"].isin(sampled_speakers) & meta["phoneme_label"].isin([p, q])
                labels = meta.loc[mask, "phoneme_label"].to_numpy()
                subset = values[mask.to_numpy()]
                if (labels == p).sum() == 0 or (labels == q).sum() == 0:
                    continue
                cp = subset[labels == p].mean(axis=0)
                cq = subset[labels == q].mean(axis=0)
                if metric == "cosine":
                    distance = float(cdist([cp], [cq], metric="cosine")[0, 0])
                elif metric == "mahalanobis":
                    distance = float(cdist([cp], [cq], metric="mahalanobis", VI=acoustic_vi)[0, 0])
                else:
                    distance = float(np.linalg.norm(cp - cq))
                boot.append(distance)
            rows.append(
                {
                    "pair": f"{p}-{q}",
                    "representation": rep_name,
                    "distance_mean": float(np.mean(boot)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "n_bootstrap": len(boot),
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "selected_pair_distance_bootstrap_ci.csv", index=False)


def nearest_centroid_classifiers(
    meta: pd.DataFrame,
    acoustic: pd.DataFrame,
    whisper: np.ndarray,
    xlsr: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    mask = meta["phoneme_label"].isin(ORAL_VOWELS).to_numpy()
    y = meta.loc[mask, "phoneme_label"].to_numpy()
    speakers = meta.loc[mask, "speaker_id"].to_numpy()
    l1_status = meta.loc[mask, "l1_status"].to_numpy()
    representations = {
        "acoustic": acoustic.loc[mask, ["f1_lobanov", "f2_lobanov"]].to_numpy(),
        "whisper_layer20": whisper[mask],
        "xlsr_layer18": xlsr[mask],
    }

    prediction_table = pd.DataFrame({"phoneme_label": y, "speaker_id": speakers, "l1_status": l1_status})
    metric_rows = []
    labels = [p for p in ORAL_VOWELS if p in set(y)]
    for name, values in representations.items():
        preds = _leave_one_speaker_out_nearest_centroid(values, y, speakers)
        prediction_table[f"pred_{name}"] = preds
        metric_rows.append(
            {
                "representation": name,
                "overall_accuracy": accuracy_score(y, preds),
                "macro_f1": f1_score(y, preds, labels=labels, average="macro", zero_division=0),
                "n_tokens": len(y),
            }
        )
        for group in ["L1", "L2"]:
            group_mask = l1_status == group
            metric_rows.append(
                {
                    "representation": name,
                    "group": group,
                    "overall_accuracy": accuracy_score(y[group_mask], preds[group_mask]),
                    "macro_f1": f1_score(y[group_mask], preds[group_mask], labels=labels, average="macro", zero_division=0),
                    "n_tokens": int(group_mask.sum()),
                }
            )
        _plot_confusion(y, preds, labels, figures_dir / f"confusion_{name}.png", title=name)

    pd.DataFrame(metric_rows).to_csv(tables_dir / "phoneme_identification_metrics.csv", index=False)
    prediction_table.to_csv(tables_dir / "phoneme_identification_predictions.csv", index=False)
    _mcnemar_comparisons(y, prediction_table, tables_dir)


def _leave_one_speaker_out_nearest_centroid(values: np.ndarray, y: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(y, dtype=object)
    for speaker in np.unique(speakers):
        test_mask = speakers == speaker
        train_mask = ~test_mask
        train_labels = y[train_mask]
        phonemes = np.array(sorted(np.unique(train_labels)))
        centroids = np.vstack([values[train_mask][train_labels == phoneme].mean(axis=0) for phoneme in phonemes])
        metric = "euclidean" if values.shape[1] == 2 else "cosine"
        distances = cdist(values[test_mask], centroids, metric=metric)
        predictions[test_mask] = phonemes[np.argmin(distances, axis=1)]
    return predictions


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(8, 7))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Leave-one-speaker-out confusion: {title}")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _mcnemar_comparisons(y: np.ndarray, predictions: pd.DataFrame, tables_dir: Path) -> None:
    names = ["acoustic", "whisper_layer20", "xlsr_layer18"]
    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            correct_a = predictions[f"pred_{name_a}"].to_numpy() == y
            correct_b = predictions[f"pred_{name_b}"].to_numpy() == y
            table = np.array(
                [
                    [np.sum(correct_a & correct_b), np.sum(correct_a & ~correct_b)],
                    [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)],
                ]
            )
            result = mcnemar(table, exact=False, correction=True)
            rows.append(
                {
                    "representation_a": name_a,
                    "representation_b": name_b,
                    "both_correct": int(table[0, 0]),
                    "a_correct_b_wrong": int(table[0, 1]),
                    "a_wrong_b_correct": int(table[1, 0]),
                    "both_wrong": int(table[1, 1]),
                    "statistic": result.statistic,
                    "p_value": result.pvalue,
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "phoneme_identification_mcnemar.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--whisper-pca", type=Path, default=Path("data/features_whisper_pca.npz"))
    parser.add_argument("--xlsr-pca", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs(args.tables_dir, args.figures_dir)

    acoustic = pd.read_csv(args.acoustic, low_memory=False).reset_index(drop=True)
    meta = acoustic[["speaker_id", "phoneme_label", "l1_status", "gender"]].copy()
    with np.load(args.whisper_pca) as whisper_data, np.load(args.xlsr_pca) as xlsr_data:
        whisper = whisper_data["pca50_layer_20"]
        xlsr = xlsr_data["pca50_layer_18"]

    acoustic_results = acoustic_group_tests(acoustic, args.tables_dir)
    gender_results = gender_residual_tests(acoustic, args.tables_dir)
    whisper_results = neural_permutation_tests(
        meta, whisper, "whisper_layer20", args.tables_dir, args.permutations, args.random_state
    )
    xlsr_results = neural_permutation_tests(
        meta, xlsr, "xlsr_layer18", args.tables_dir, args.permutations, args.random_state
    )
    distance_matrices(meta, acoustic, whisper, xlsr, args.tables_dir)
    nearest_centroid_classifiers(meta, acoustic, whisper, xlsr, args.tables_dir, args.figures_dir)

    summary = {
        "acoustic_fdr_significant": int(acoustic_results["significant_fdr_0_05"].sum()),
        "gender_residual_fdr_significant": int(gender_results["significant_fdr_0_05"].sum()),
        "whisper_fdr_significant": int(whisper_results["significant_fdr_0_05"].sum()),
        "xlsr_fdr_significant": int(xlsr_results["significant_fdr_0_05"].sum()),
    }
    print(summary)


if __name__ == "__main__":
    main()
