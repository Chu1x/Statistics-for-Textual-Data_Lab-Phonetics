#!/usr/bin/env python3
"""Write a compact Markdown report from generated result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


TABLES = Path("results/tables")
DATA = Path("data")
CONFIG = Path("config/config.yaml")
ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def _weighted_missingness(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for column in columns:
        rows.append(
            {
                "feature": column,
                "n_tokens_considered": total,
                "missing_prop": float(df[column].isna().mean()) if total else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _load_config() -> dict[str, object]:
    with CONFIG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _table(df: pd.DataFrame) -> str:
    visible = df.copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(lambda value: _fmt(value))
    lines = []
    for row in visible.to_dict(orient="records"):
        parts = []
        for key, value in row.items():
            if pd.isna(value):
                value = "overall"
            parts.append(f"{key}: {value}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)


def _long_table(df: pd.DataFrame, row_prefix: str = "row") -> str:
    visible = df.copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(lambda value: _fmt(value))
    lines = []
    for row_number, row in enumerate(visible.to_dict(orient="records"), start=1):
        lines.append(f"- **{row_prefix} {row_number}**")
        for key, value in row.items():
            if pd.isna(value):
                value = "overall"
            lines.append(f"  - `{key}`: {value}")
    return "\n".join(lines)


def _short_label(value: object) -> str:
    text = str(value)
    replacements = {
        "whisper_layer20": "Whisper",
        "xlsr_layer18": "XLS-R",
        "acoustic": "Acoustic",
        "acoustic_euclidean": "Acoustic",
        "whisper_cosine": "Whisper",
        "xlsr_cosine": "XLS-R",
        "p_value_asymptotic": "p",
    }
    return replacements.get(text, text)


def _compact_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        row_dict = row._asdict()
        for column, _ in columns:
            value = row_dict[column]
            if pd.isna(value):
                value = "all"
            elif isinstance(value, float):
                value = _fmt(value)
            else:
                value = _short_label(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _file_index(rows: list[tuple[str, str]]) -> str:
    lines = []
    for path, description in rows:
        lines.append(f"- `{path}`")
        lines.append(f"  - {description}")
    return "\n".join(lines)


def _path_list(paths: list[str]) -> str:
    return "\n".join(f"- `{path}`" for path in paths)


def build_report() -> str:
    config = _load_config()
    tokens = pd.read_csv(DATA / "phoneme_tokens.csv", usecols=["speaker_id", "sentence_id"])
    acoustic_missingness_by_group = pd.read_csv(DATA / "features_acoustic_missingness.csv")
    acoustic_missingness_by_group.to_csv(TABLES / "acoustic_missingness_by_phoneme_group.csv", index=False)
    acoustic_norm = pd.read_csv(
        DATA / "features_acoustic_norm.csv",
        usecols=[
            "phoneme_label",
            "is_fricative",
            "duration_ms",
            "f1_hz",
            "f2_hz",
            "f3_hz",
            "f0_mean_hz",
            "scg_hz",
            "f1_25_hz",
            "f2_25_hz",
            "f1_75_hz",
            "f2_75_hz",
        ],
        low_memory=False,
    )
    acoustic_tests = _read("acoustic_l1_l2_tests.csv")
    acoustic_sig = acoustic_tests[acoustic_tests["significant_fdr_0_05"]]
    acoustic_quality = _read("acoustic_quality_flags.csv")
    acoustic_sensitivity = _read("acoustic_l1_l2_sensitivity_range_filter.csv")
    gender_tests = _read("gender_residual_tests.csv")
    whisper_perm = _read("neural_l1_l2_permutation_whisper_layer20.csv")
    xlsr_perm = _read("neural_l1_l2_permutation_xlsr_layer18.csv")
    trajectory = _read("trajectory_midpoint_comparison.csv")
    neural_metrics = _read("neural_projection_metrics.csv")
    rsm = _read("rsm_mantel_sample.csv")
    distance_mantel = _read("phoneme_distance_mantel.csv")
    pair_bootstrap = _read("selected_pair_distance_bootstrap_ci.csv")
    identification = _read("phoneme_identification_metrics.csv")
    mcnemar = _read("phoneme_identification_mcnemar.csv")
    model_comparisons = _read("mixed_model_comparisons.csv")
    fixed_effects = _read("mixed_model_fixed_effects.csv")
    icc = _read("mixed_model_icc_a.csv")
    r2 = _read("mixed_model_representation_r2_summary.csv")
    rope = _read("rope_summary.csv")
    vowel_clustering = _read("clustering_vowel_ari.csv")
    speaker_clustering = _read("clustering_speaker_ari.csv")
    cv_clustering = _read("clustering_consonant_vowel_ari.csv")
    misclustered = _read("clustering_consonant_vowel_misclustered.csv")

    best_identification = identification[identification["group"].isna()].sort_values(
        "overall_accuracy", ascending=False
    ).iloc[0]
    best_height = vowel_clustering.sort_values("ari_height_k3", ascending=False).iloc[0]
    best_front_back = vowel_clustering.sort_values("ari_front_back_central_k4", ascending=False).iloc[0]
    best_cv = cv_clustering.sort_values("ari_consonant_vowel", ascending=False).iloc[0]
    best_r2 = r2.sort_values("mean_marginal_r2", ascending=False).iloc[0]
    gender_sig_count = int(gender_tests["significant_fdr_0_05"].sum())
    whisper_sig_count = int(whisper_perm["significant_fdr_0_05"].sum())
    xlsr_sig_count = int(xlsr_perm["significant_fdr_0_05"].sum())
    whisper_total = len(whisper_perm)
    xlsr_total = len(xlsr_perm)
    acoustic_total = len(acoustic_tests)
    trajectory_changed = int(trajectory["conclusion_changed"].sum())
    trajectory_total = len(trajectory)
    trajectory_biggest = trajectory.sort_values("absolute_difference_change", ascending=False).head(5)
    pair_bootstrap_display = pair_bootstrap.head(8).copy()
    mcnemar_display = mcnemar.copy()
    mcnemar_sig_count = int((mcnemar["p_value"] < 0.05).sum())
    model_lrt_display = model_comparisons.loc[
        model_comparisons["comparison"].notna(),
        ["representation", "response", "comparison", "lr_statistic", "raw_lr_statistic", "lrt_note", "df", "p_value"],
    ].head(12).copy()
    fixed_effects_display = fixed_effects.head(10).copy()
    vowel_tokens = acoustic_norm[acoustic_norm["phoneme_label"].isin(ORAL_VOWELS)]
    fricative_tokens = acoustic_norm[acoustic_norm["is_fricative"].astype(bool)]
    long_vowel_tokens = vowel_tokens[vowel_tokens["duration_ms"] > 80]
    missingness = pd.concat(
        [
            _weighted_missingness(vowel_tokens, ["f1_hz", "f2_hz", "f3_hz", "f0_mean_hz"]),
            _weighted_missingness(fricative_tokens, ["scg_hz"]),
            _weighted_missingness(long_vowel_tokens, ["f1_25_hz", "f2_25_hz", "f1_75_hz", "f2_75_hz"]),
        ],
        ignore_index=True,
    )
    quality_display = acoustic_quality[
        acoustic_quality["subset"].isin(["oral_vowels", "all_tokens"])
        & acoustic_quality["feature"].isin(["f1_hz", "f2_hz", "f0_mean_hz"])
    ].copy()
    oral_quality = acoustic_quality[acoustic_quality["subset"].eq("oral_vowels")]
    oral_f1_flags = int(oral_quality.loc[oral_quality["feature"].eq("f1_hz"), "n_flagged"].iloc[0])
    oral_f2_flags = int(oral_quality.loc[oral_quality["feature"].eq("f2_hz"), "n_flagged"].iloc[0])
    oral_f0_flags = int(oral_quality.loc[oral_quality["feature"].eq("f0_mean_hz"), "n_flagged"].iloc[0])
    sensitivity_changed = int(acoustic_sensitivity["conclusion_changed"].sum())
    sensitivity_total = len(acoustic_sensitivity)
    sensitivity_display = acoustic_sensitivity[
        acoustic_sensitivity["n_excluded_by_range_filter"].fillna(0).gt(0)
        | acoustic_sensitivity["conclusion_changed"]
    ].copy()
    sensitivity_display = sensitivity_display.sort_values(
        ["conclusion_changed", "absolute_effect_change"], ascending=[False, False]
    ).head(8)
    n_tokens = len(tokens)
    n_speakers = tokens["speaker_id"].nunique()
    n_sentences = tokens["sentence_id"].nunique()
    whisper_layers = ", ".join(str(layer) for layer in config["layers"]["whisper"])
    xlsr_layers = ", ".join(str(layer) for layer in config["layers"]["xlsr"])

    neural_metrics_display = neural_metrics.sort_values(
        "between_phoneme_variance_ratio_2d", ascending=False
    ).head(5).copy()

    acoustic_sig_list = ", ".join(
        f"/{row.phoneme_label}/ {row.feature.replace('_lobanov', '')}"
        for row in acoustic_sig.itertuples(index=False)
    )
    neural_whisper_sig_list = ", ".join(
        f"/{row.phoneme_label}/"
        for row in whisper_perm[whisper_perm["significant_fdr_0_05"]].itertuples(index=False)
    )
    neural_xlsr_sig_list = ", ".join(
        f"/{row.phoneme_label}/"
        for row in xlsr_perm[xlsr_perm["significant_fdr_0_05"]].itertuples(index=False)
    )
    rope_counts = rope.groupby(["representation", "rope_classification"]).size().reset_index(name="n")
    rope_lines = "\n".join(
        f"- {row.representation}: {row.rope_classification} = {row.n}"
        for row in rope_counts.itertuples(index=False)
    )
    supplementary_outputs = [
        ("results/tables/acoustic_missingness_by_phoneme_group.csv", "missing acoustic-value proportions by phoneme, L1/L2 group and gender."),
        ("results/tables/acoustic_vowel_descriptives.csv", "complete vowel-level acoustic descriptives by speaker group, including mean, median, SD, IQR and CV for F1/F2."),
        ("results/tables/acoustic_f1_variance_decomposition.csv", "per-vowel decomposition of F1 variation into total, inter-speaker and intra-speaker components."),
        ("results/tables/acoustic_l1_l2_tests.csv", "full acoustic L1/L2 vowel tests with raw p-values and BH-FDR adjusted p-values."),
        ("results/tables/gender_residual_tests.csv", "speaker-level residual gender tests after Lobanov normalisation."),
        ("results/tables/neural_l1_l2_permutation_whisper_layer20.csv", "Whisper layer 20 L1/L2 permutation tests for vowel contrasts."),
        ("results/tables/neural_l1_l2_permutation_xlsr_layer18.csv", "XLS-R layer 18 L1/L2 permutation tests for vowel contrasts."),
        ("results/tables/neural_projection_metrics.csv", "PCA/UMAP projection diagnostics for all tested neural layers."),
        ("results/tables/rsm_mantel_sample.csv", "sampled representational similarity correlations across acoustic, Whisper and XLS-R token distances."),
        ("results/tables/phoneme_distance_mantel.csv", "Mantel correlations between phoneme-centroid distance matrices."),
        ("results/tables/selected_pair_distance_bootstrap_ci.csv", "speaker-level bootstrap confidence intervals for selected phoneme-pair distances."),
        ("results/tables/phoneme_identification_metrics.csv", "overall and group-specific nearest-centroid identification metrics."),
        ("results/tables/phoneme_identification_mcnemar.csv", "matched-pair McNemar comparisons between identification systems."),
        ("results/tables/mixed_model_comparisons.csv", "null, main-effects, interaction, extended and likelihood-ratio model-comparison sequence."),
        ("results/tables/mixed_model_fixed_effects.csv", "fixed-effect estimates from the extended mixed-effects models."),
        ("results/tables/mixed_model_icc_a.csv", "speaker ICC estimates for /a/ in acoustic, Whisper and XLS-R responses."),
        ("results/tables/mixed_model_random_slope_note.csv", "documentation of why the by-speaker L1 random slope was not fitted."),
        ("results/tables/mixed_model_representation_r2_summary.csv", "marginal and conditional R2 summaries by representation."),
        ("results/tables/rope_acoustic_contrasts.csv", "full acoustic ROPE contrast table for F1/F2 in Hz."),
        ("results/tables/rope_neural_contrasts_whisper_layer20.csv", "Whisper layer 20 contrast-level ROPE classifications."),
        ("results/tables/rope_neural_contrasts_xlsr_layer18.csv", "XLS-R layer 18 contrast-level ROPE classifications."),
        ("results/tables/rope_summary.csv", "combined ROPE summary used in the main text."),
        ("results/tables/clustering_vowel_ari.csv", "vowel clustering metrics, linkage choices, silhouettes and ARI values."),
        ("results/tables/clustering_consonant_vowel_ari.csv", "consonant/vowel clustering ARI and silhouette metrics."),
        ("results/tables/clustering_speaker_ari.csv", "speaker clustering results against L1/L2 and gender labels."),
    ]
    projection_figures = _path_list(
        [
            f"results/figures/{name}"
            for name in [
            "whisper_layer_6_pca2.png",
            "whisper_layer_6_umap2.png",
            "whisper_layer_20_pca2.png",
            "whisper_layer_20_umap2.png",
            "xlsr_layer_3_pca2.png",
            "xlsr_layer_3_umap2.png",
            "xlsr_layer_9_pca2.png",
            "xlsr_layer_9_umap2.png",
            "xlsr_layer_18_pca2.png",
            "xlsr_layer_18_umap2.png",
        ]
        ]
    )
    systematic = misclustered.sort_values("n_binary_misclustered", ascending=False).head(3)
    systematic_labels = ", ".join(f"/{row.phoneme_label}/" for row in systematic.itertuples(index=False))

    return f"""# Acoustic and Neural Representations in a Phonetically Aligned Corpus

## Overview

This project asks how far acoustic measurements and neural speech representations tell the same story about phonetic structure in the Russian-French Interference Corpus. I treated the corpus as a phoneme-level dataset: TextGrid intervals were converted into token rows, acoustic descriptors were measured at aligned intervals, Whisper and XLS-R hidden states were averaged over the same intervals, and the resulting representations were compared with descriptive, inferential, and clustering analyses.

## Pipeline Outputs

Main data products:

- `data/phoneme_tokens.csv`
- `data/features_acoustic.csv`
- `data/features_acoustic_norm.csv`
- `data/features_whisper.npz`
- `data/features_whisper_pca.npz`
- `data/features_xlsr.npz`
- `data/features_xlsr_pca.npz`

Main result folders:

- `results/tables/`
- `results/figures/`

## Methods and Data Quality

After parsing, the working dataset contained {n_tokens} phoneme tokens from {n_speakers} speakers and {n_sentences} sentence IDs. Acoustic extraction used Praat/parselmouth Burg LPC formants with `max_number_of_formants = 5`. The maximum formant was 5000 Hz for female speakers and 4500 Hz for male speakers. F1-F3 were measured at each phoneme midpoint; for vowels longer than 80 ms, F1-F3 were also measured at 25% and 75% of the interval. Formants were Lobanov-normalised within speaker, using vowel tokens only, so that speaker differences in vocal-tract scale did not dominate the vowel comparisons.

For the neural analyses, I extracted `{config["models"]["whisper"]}` layers {whisper_layers} and `{config["models"]["xlsr"]}` layers {xlsr_layers}. Each selected layer was reduced to {config["runtime"]["pca_components"]} principal components before downstream modelling. Permutation tests used {config["runtime"]["permutations"]} permutations, and bootstrap/ROPE summaries used {config["runtime"]["bootstrap"]} resamples where applicable.

Missing-value summary by analysis-relevant token set:

{_long_table(missingness, "missingness")}

Rough acoustic range flags:

{_long_table(quality_display, "quality")}

I did not impute missing acoustic values. Instead, each analysis uses the available tokens for the feature being tested, and the corresponding result tables report the relevant sample sizes. Rough-range values were kept in the main analysis but tracked as quality diagnostics; among oral vowels, the flagged counts were F1 = {oral_f1_flags}, F2 = {oral_f2_flags}, and f0 = {oral_f0_flags}.

Missingness by phoneme class and speaker group is reported in `results/tables/acoustic_missingness_by_phoneme_group.csv`.

As a robustness check, I reran the acoustic L1/L2 tests after excluding rough-range F1/F2 values. This changed {sensitivity_changed} of {sensitivity_total} FDR-significance decisions. The affected or filtered contrasts were:

{_long_table(sensitivity_display, "sensitivity")}

## Supplementary outputs and result index

The PDF report gives the main results needed to answer the project questions. Full result tables and supporting figures are included in the submitted folder so that every reported decision can be checked without relying on prose alone.

{_file_index(supplementary_outputs)}

## Descriptive Statistics

The descriptive stage is documented in three layers. The main text reports the most important summaries; the full supporting files are:

- `results/tables/acoustic_vowel_descriptives.csv`
  - Full mean, median, SD, IQR and CV by vowel and speaker group.
- `results/tables/acoustic_f1_variance_decomposition.csv`
  - F1 variance decomposition by vowel.
- `results/figures/intra_speaker_variability_violin.png`
  - Intra-speaker variability figure.

For neural representations, `results/tables/neural_projection_metrics.csv` gives PCA/UMAP diagnostics for all tested layers. The submitted projection figures are:

{projection_figures}

The largest 2D between-phoneme variance ratios were:

{_long_table(neural_metrics_display, "projection")}

Sampled RSM correlations:

{_long_table(rsm, "rsm")}

## Statistical Tests

The complete vowel-level acoustic tests are in `results/tables/acoustic_l1_l2_tests.csv`; this table includes the test used for each contrast and the BH-FDR correction. After BH-FDR correction, {len(acoustic_sig)} of {acoustic_total} acoustic vowel-feature contrasts were significant: {acoustic_sig_list}.

Neural L1/L2 permutation tests are reported in `results/tables/neural_l1_l2_permutation_whisper_layer20.csv` and `results/tables/neural_l1_l2_permutation_xlsr_layer18.csv`. Whisper layer 20 had {whisper_sig_count} of {whisper_total} vowel contrasts significant after BH-FDR correction ({neural_whisper_sig_list}); XLS-R layer 18 also had {xlsr_sig_count} of {xlsr_total} significant contrasts ({neural_xlsr_sig_list}).

Phoneme-centroid distance Mantel correlations:

{_long_table(distance_mantel, "distance")}

Residual gender-effect tests after Lobanov normalisation are reported in `results/tables/gender_residual_tests.csv`. They found {gender_sig_count} FDR-significant acoustic contrasts at alpha = 0.05.

## Midpoint vs Trajectory

For long vowels with available 25% and 75% formant measurements, midpoint-based L1/L2 conclusions were compared with trajectory-mean conclusions on the same subset of tokens. {trajectory_changed} of {trajectory_total} FDR-significance decisions changed.

Largest changes in L2-L1 effect size:

{_long_table(trajectory_biggest, "trajectory")}

## Inter-phoneme Distances and Classification

The Mantel summaries above compare complete phoneme-centroid distance matrices. `results/tables/selected_pair_distance_bootstrap_ci.csv` adds speaker-level bootstrap confidence intervals for selected phoneme pairs; the first rows are:

{_long_table(pair_bootstrap_display, "pair")}

Nearest-centroid classification results are in `results/tables/phoneme_identification_metrics.csv`. The table includes overall metrics and, where present, separate L1/L2 rows:

{_long_table(identification, "classifier")}

The best phoneme identification accuracy was obtained by `{best_identification.representation}` with accuracy {_fmt(best_identification.overall_accuracy)}.

Matched-pair classifier comparisons are reported in `results/tables/phoneme_identification_mcnemar.csv`. In that table, {mcnemar_sig_count} of {len(mcnemar)} McNemar tests have p < 0.05:

{_long_table(mcnemar_display, "mcnemar")}

## Mixed-Effects Models

`results/tables/mixed_model_comparisons.csv` reports the null, main-effects, full-interaction and extended model-building sequence, together with likelihood-ratio comparisons. These nested model comparisons were fitted with maximum likelihood rather than REML in the analysis code. A compact excerpt is:

{_long_table(model_lrt_display, "model")}

`results/tables/mixed_model_fixed_effects.csv` reports fixed-effect estimates from the extended models. The first rows are:

{_long_table(fixed_effects_display, "fixed")}

`results/tables/mixed_model_random_slope_note.csv` documents the random-slope decision: L1/L2 status is constant within speaker, so a by-speaker random slope for L1 has no within-speaker variation.

ICC for /a/ is reported in `results/tables/mixed_model_icc_a.csv`:

{_long_table(icc, "icc")}

Marginal and conditional R2 are summarised in `results/tables/mixed_model_representation_r2_summary.csv`:

{_long_table(r2, "r2")}

The highest mean marginal R2 was obtained by `{best_r2.representation}` with mean marginal R2 {_fmt(best_r2.mean_marginal_r2)}.

## Confidence Intervals and ROPE

The full contrast-level ROPE classifications are in `results/tables/rope_acoustic_contrasts.csv`, `results/tables/rope_neural_contrasts_whisper_layer20.csv`, `results/tables/rope_neural_contrasts_xlsr_layer18.csv`, and the combined `results/tables/rope_summary.csv`. The corresponding forest plots are `results/figures/forest_acoustic_f1_rope.png`, `results/figures/forest_acoustic_f2_rope.png`, `results/figures/forest_whisper_layer20_rope.png`, and `results/figures/forest_xlsr_layer18_rope.png`.

ROPE classification counts:

{rope_lines}

Acoustic F1 used the default [-20, +20] Hz ROPE. Neural ROPEs used the intra-speaker cosine-distance noise floor. The acoustic CI implementation is a speaker-level interval approximation rather than a strict profile-likelihood interval, so ROPE classifications should be interpreted as transparent robustness summaries rather than exact profile-likelihood decisions.

## Hierarchical Clustering

Acoustic clustering used Euclidean distances with Ward linkage. Neural clustering used cosine distances with average linkage, as recorded in the clustering tables. This is a deliberate deviation from Ward for the neural spaces: Ward's criterion is Euclidean, so average linkage was used for cosine-based neural distances.

Vowel clustering:

{_long_table(vowel_clustering, "vowel")}

Consonant/vowel clustering:

{_long_table(cv_clustering, "cv")}

Speaker clustering:

{_long_table(speaker_clustering, "speaker")}

Best vowel-height recovery: `{best_height.representation}` with ARI {_fmt(best_height.ari_height_k3)}.
Best front/back/central recovery: `{best_front_back.representation}` with ARI {_fmt(best_front_back.ari_front_back_central_k4)}.
Best consonant/vowel boundary recovery: `{best_cv.representation}` with ARI {_fmt(best_cv.ari_consonant_vowel)}.

Systematically difficult consonants included {systematic_labels}; the supporting counts are in `results/tables/clustering_consonant_vowel_misclustered.csv`.

## Answers to the 16 Questions

1. PCA and UMAP were useful in different ways. PCA gave a stable linear baseline, and the strongest 2D separation appeared in the higher XLS-R layers. UMAP made local phoneme neighbourhoods easier to see, but I do not interpret its axes as acoustic dimensions.
2. The acoustic summaries point to /ɑ/, /u/, /y/, and /ɛ/ as especially variable categories. The neural spaces show related structure, but not a one-to-one copy of the formant space, which is unsurprising because the embeddings also carry context and speaker information.
3. The UMAP plots show clear phoneme neighbourhoods, particularly for XLS-R, but they do not reproduce the IPA vowel trapezoid exactly. In other words, the neural space is phonetically organised, but not simply an F1/F2 chart in another form.
4. The RSM results support the same interpretation. Acoustic-XLS-R similarity was higher than acoustic-Whisper similarity, r = 0.355 versus r = 0.189, while Whisper and XLS-R were more similar to each other, r = 0.676.
5. In the acoustic tests, 9 contrasts survived FDR correction, involving /i/, /y/, /u/, /ø/, /ɛ/, and /ɑ/. Whisper and XLS-R each had 9 significant vowel-level neural contrasts after FDR correction. The main exception was /œ/, where the corpus simply does not provide enough data, and /ə/, which was not significant.
6. Euclidean and Mahalanobis acoustic distances gave almost the same vowel-distance ranking, r = 0.984. The acoustic-neural correlations were moderate and very similar for Whisper and XLS-R, while the two neural distance matrices were more closely aligned with each other, r = 0.842.
7. Whisper layer 20 was the best phoneme classifier in the leave-one-speaker-out test, with accuracy {_fmt(best_identification.overall_accuracy)} and macro-F1 0.735. XLS-R layer 18 followed at 0.790 accuracy, and the acoustic baseline reached 0.687. `results/tables/phoneme_identification_mcnemar.csv` reports matched-pair McNemar tests for these classifier differences; all three pairwise p-values in that table are below 0.05.
8. The /a/ models show only modest speaker-specificity: ICC = 0.038 for acoustic F1, 0.042 for Whisper PC1, and 0.031 for XLS-R PC1. Most of the remaining variation is therefore token-level or residual rather than stable between-speaker variation.
9. The gender results depend on which analysis is used. The speaker-level residual tests after Lobanov normalisation found 0 FDR-significant acoustic contrasts, so there is no broad residual gender pattern in `results/tables/gender_residual_tests.csv`. The mixed-model comparison table still reports some significant L1-by-gender interaction tests for acoustic responses, so I treat this as a model-specific interaction rather than as a simple across-vowel residual gender effect. In the fixed-effect estimates, the clearest neural interaction noted in the report is XLS-R PC2, estimate = 6.649, p = 0.031.
10. The mixed-effects models gave the highest mean marginal R2 to the acoustic features, {_fmt(best_r2.mean_marginal_r2)}. XLS-R layer 18 was close behind at 0.410, while Whisper layer 20 was lower at 0.310. This fits the idea that formants are more directly tied to the fixed vowel predictors, whereas neural PCs contain additional information.
11. I did not find a case where a statistically significant acoustic F1 effect was also clearly inside the acoustic ROPE. The acoustic ROPE results are therefore better read as a mix of non-equivalent and indeterminate contrasts, rather than as evidence for practically negligible formant differences.
12. The ROPE results differ quite sharply across representations. Acoustic F1 produced 2 non-equivalent, 8 indeterminate, and 1 insufficient contrast. Whisper produced 10 equivalent contrasts and 1 insufficient case. XLS-R sat between them, with 6 equivalent, 4 indeterminate, and 1 insufficient contrast.
13. This disagreement is meaningful rather than contradictory. Formants isolate narrow articulatory differences, while neural cosine distances pool phonetic detail with context, speaker, and model-specific information. A contrast can therefore be acoustically robust while still falling within the neural noise floor.
14. The clustering results split the phonological structure across models. Whisper layer 20 recovered vowel height best, ARI = 0.499. XLS-R layer 18 recovered front/back/central grouping best, ARI = 0.305. The acoustic representation had the clearest vowel-clustering silhouette, but not the strongest height ARI.
15. Speaker clustering revealed strong group information in the neural spaces. Whisper layer 20 separated L1/L2 perfectly at k = 2, ARI = 1.000, while XLS-R layer 18 separated gender perfectly, ARI = 1.000. Because this speaker-level analysis has only 19 speakers, these perfect ARI values should be treated as descriptive evidence rather than a large-sample generalisation.
16. For consonant/vowel clustering, XLS-R layer 18 performed best, ARI = 0.638, followed by the acoustic representation at 0.487. Whisper layer 20 was near chance, ARI = -0.024. The recurrently difficult consonants were /l/, /n/, and /ʁ/, which makes phonetic sense because sonorants and rhotics sit closer to vowels than obstruents do.

## Limitations

- The corpus contains {n_speakers} speakers, so speaker-level clustering and mixed-effects summaries should be interpreted cautiously.
- RSM computation used a sampled token set, as reported in `results/tables/rsm_mantel_sample.csv`, rather than all possible corpus tokens.
- Acoustic ROPE intervals are speaker-level approximations rather than strict profile-likelihood confidence intervals.
- Neural clustering used average linkage for cosine distances because Ward linkage is tied to Euclidean geometry.
- Perfect ARI values in speaker clustering are descriptive and should be interpreted cautiously because `n_speakers = {n_speakers}`.
- Missingness and acoustic quality handling rely on available-case analysis and rough range flags, not imputation or manual correction.

## Reproducibility

Run the full workflow with:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake --cores 1
```

Dry-run the workflow with:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake -n --cores 1
```

The workflow parameters are stored in `config/config.yaml`.

## References

- Lobanov, B. M. (1971). Classification of Russian vowels spoken by different speakers.
- Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations.
- Conneau, A. et al. (2021). Unsupervised cross-lingual representation learning for speech recognition.
- Radford, A. et al. (2023). Robust speech recognition via large-scale weak supervision.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection.
- Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation.

\\clearpage
\\begingroup
\\setlength{{\\intextsep}}{{0.25em}}
\\setlength{{\\textfloatsep}}{{0.25em}}
\\setlength{{\\floatsep}}{{0.25em}}
\\setlength{{\\abovecaptionskip}}{{0.15em}}
\\setlength{{\\belowcaptionskip}}{{0.15em}}

## Figures

Figures are grouped here so that they do not interrupt the long result listings in the main text.

\\Needspace{{0.42\\textheight}}
**Figure 1. F1 by vowel and L1/L2 group**

![](figures/f1_lobanov_boxplot_by_group.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 2. F2 by vowel and L1/L2 group**

![](figures/f2_lobanov_boxplot_by_group.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 3. Lobanov vowel chart**

![](figures/vowel_chart_lobanov.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 4. XLS-R layer 18 UMAP projection**

![](figures/xlsr_layer_18_umap2.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 5. Midpoint vs 25/75 trajectory mean**

![](figures/trajectory_midpoint_vs_trajectory.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 6. Whisper layer 20 confusion matrix**

![](figures/confusion_whisper_layer20.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 7. Acoustic F1 ROPE forest plot**

![](figures/forest_acoustic_f1_rope.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 8. XLS-R consonant/vowel dendrogram**

![](figures/dendrogram_consonant_vowel_xlsr_layer18.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 9. Whisper layer 20 speaker dendrogram**

![](figures/dendrogram_speakers_whisper_layer20.png){{width=65%}}
\\vspace{{-0.8em}}

\\Needspace{{0.42\\textheight}}
**Figure 10. XLS-R layer 18 speaker dendrogram**

![](figures/dendrogram_speakers_xlsr_layer18.png){{width=65%}}

\\endgroup
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
