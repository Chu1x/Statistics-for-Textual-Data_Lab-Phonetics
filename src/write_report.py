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


def build_report() -> str:
    config = _load_config()
    tokens = pd.read_csv(DATA / "phoneme_tokens.csv", usecols=["speaker_id", "sentence_id"])
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
    trajectory = _read("trajectory_midpoint_comparison.csv")
    neural_metrics = _read("neural_projection_metrics.csv")
    rsm = _read("rsm_mantel_sample.csv")
    distance_mantel = _read("phoneme_distance_mantel.csv")
    identification = _read("phoneme_identification_metrics.csv")
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
    trajectory_changed = int(trajectory["conclusion_changed"].sum())
    trajectory_total = len(trajectory)
    trajectory_biggest = trajectory.sort_values("absolute_difference_change", ascending=False).head(5)
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
    rope_counts = rope.groupby(["representation", "rope_classification"]).size().reset_index(name="n")
    rope_lines = "\n".join(
        f"- {row.representation}: {row.rope_classification} = {row.n}"
        for row in rope_counts.itertuples(index=False)
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

After parsing, the working dataset contained {n_tokens} phoneme tokens from {n_speakers} speakers and {n_sentences} sentence IDs. Acoustic extraction used midpoint formants and, for vowels longer than 80 ms, additional 25% and 75% formant measurements. Formants were Lobanov-normalised within speaker, using vowel tokens only, so that speaker differences in vocal-tract scale did not dominate the vowel comparisons.

For the neural analyses, I extracted `{config["models"]["whisper"]}` layers {whisper_layers} and `{config["models"]["xlsr"]}` layers {xlsr_layers}. Each selected layer was reduced to {config["runtime"]["pca_components"]} principal components before downstream modelling. Permutation tests used {config["runtime"]["permutations"]} permutations, and bootstrap/ROPE summaries used {config["runtime"]["bootstrap"]} resamples where applicable.

Missing-value summary by analysis-relevant token set:

{_long_table(missingness, "missingness")}

Rough acoustic range flags:

{_long_table(quality_display, "quality")}

I did not impute missing acoustic values. Instead, each analysis uses the available tokens for the feature being tested, and the corresponding result tables report the relevant sample sizes. Rough-range values were kept in the main analysis but tracked as quality diagnostics; among oral vowels, the flagged counts were F1 = {oral_f1_flags}, F2 = {oral_f2_flags}, and f0 = {oral_f0_flags}.

As a robustness check, I reran the acoustic L1/L2 tests after excluding rough-range F1/F2 values. This changed {sensitivity_changed} of {sensitivity_total} FDR-significance decisions. The affected or filtered contrasts were:

{_long_table(sensitivity_display, "sensitivity")}

## Descriptive Statistics

The largest 2D between-phoneme variance ratios were observed for:

{_long_table(neural_metrics_display, "projection")}

Sampled RSM correlations:

{_long_table(rsm, "rsm")}

## Statistical Tests

After BH-FDR correction, acoustic L1/L2 differences persisted for: {acoustic_sig_list}.

Phoneme-centroid distance Mantel correlations:

{_long_table(distance_mantel, "distance")}

Residual gender-effect tests after Lobanov normalisation found {gender_sig_count} FDR-significant contrasts at alpha = 0.05.

## Midpoint vs Trajectory

For long vowels with available 25% and 75% formant measurements, midpoint-based L1/L2 conclusions were compared with trajectory-mean conclusions on the same subset of tokens. {trajectory_changed} of {trajectory_total} FDR-significance decisions changed.

Largest changes in L2-L1 effect size:

{_long_table(trajectory_biggest, "trajectory")}

Nearest-centroid classification:

{_long_table(identification, "classifier")}

The best phoneme identification accuracy was obtained by `{best_identification.representation}` with accuracy {_fmt(best_identification.overall_accuracy)}.

## Mixed-Effects Models

ICC for /a/:

{_long_table(icc, "icc")}

Marginal and conditional R2 summary:

{_long_table(r2, "r2")}

The highest mean marginal R2 was obtained by `{best_r2.representation}` with mean marginal R2 {_fmt(best_r2.mean_marginal_r2)}.

## Confidence Intervals and ROPE

ROPE classification counts:

{rope_lines}

Acoustic F1 used the default [-20, +20] Hz ROPE. Neural ROPEs used the intra-speaker cosine-distance noise floor. The acoustic CI implementation is a speaker-level interval approximation rather than a strict profile-likelihood interval, so ROPE classifications should be interpreted as transparent robustness summaries rather than exact profile-likelihood decisions.

## Hierarchical Clustering

Vowel clustering:

{_long_table(vowel_clustering, "vowel")}

Consonant/vowel clustering:

{_long_table(cv_clustering, "cv")}

Speaker clustering:

{_long_table(speaker_clustering, "speaker")}

Best vowel-height recovery: `{best_height.representation}` with ARI {_fmt(best_height.ari_height_k3)}.
Best front/back/central recovery: `{best_front_back.representation}` with ARI {_fmt(best_front_back.ari_front_back_central_k4)}.
Best consonant/vowel boundary recovery: `{best_cv.representation}` with ARI {_fmt(best_cv.ari_consonant_vowel)}.

Systematically difficult consonants included {systematic_labels}, consistent with sonorants behaving as acoustically intermediate categories.

## Answers to the 16 Questions

1. PCA and UMAP were useful in different ways. PCA gave a stable linear baseline, and the strongest 2D separation appeared in the higher XLS-R layers. UMAP made local phoneme neighbourhoods easier to see, but I do not interpret its axes as acoustic dimensions.
2. The acoustic summaries point to /ɑ/, /u/, /y/, and /ɛ/ as especially variable categories. The neural spaces show related structure, but not a one-to-one copy of the formant space, which is unsurprising because the embeddings also carry context and speaker information.
3. The UMAP plots show clear phoneme neighbourhoods, particularly for XLS-R, but they do not reproduce the IPA vowel trapezoid exactly. In other words, the neural space is phonetically organised, but not simply an F1/F2 chart in another form.
4. The RSM results support the same interpretation. Acoustic-XLS-R similarity was higher than acoustic-Whisper similarity, r = 0.355 versus r = 0.189, while Whisper and XLS-R were more similar to each other, r = 0.676.
5. In the acoustic tests, 9 contrasts survived FDR correction, involving /i/, /y/, /u/, /ø/, /ɛ/, and /ɑ/. Whisper and XLS-R also showed widespread L1/L2 separation, with 9 significant vowel-level neural contrasts each. The main exception was /œ/, where the corpus simply does not provide enough data, and /ə/, which was not significant.
6. Euclidean and Mahalanobis acoustic distances gave almost the same vowel-distance ranking, r = 0.984. The acoustic-neural correlations were moderate and very similar for Whisper and XLS-R, while the two neural distance matrices were more closely aligned with each other, r = 0.842.
7. Whisper layer 20 was the best phoneme classifier in the leave-one-speaker-out test, with accuracy {_fmt(best_identification.overall_accuracy)} and macro-F1 0.735. XLS-R layer 18 followed at 0.790 accuracy, and the acoustic baseline reached 0.687. The McNemar comparisons suggest that these differences are not just noise.
8. The /a/ models show only modest speaker-specificity: ICC = 0.038 for acoustic F1, 0.042 for Whisper PC1, and 0.031 for XLS-R PC1. Most of the remaining variation is therefore token-level or residual rather than stable between-speaker variation.
9. I did not find a broad L1-by-gender pattern. The interaction was not significant for acoustic F1/F2 or for Whisper PCs 1-5. The one exception was XLS-R PC2, estimate = 6.649, p = 0.031, which suggests a local gender-related effect in that neural dimension.
10. The mixed-effects models gave the highest mean marginal R2 to the acoustic features, {_fmt(best_r2.mean_marginal_r2)}. XLS-R layer 18 was close behind at 0.410, while Whisper layer 20 was lower at 0.310. This fits the idea that formants are more directly tied to the fixed vowel predictors, whereas neural PCs contain additional information.
11. I did not find a case where a statistically significant acoustic F1 effect was also clearly inside the acoustic ROPE. The acoustic ROPE results are therefore better read as a mix of non-equivalent and indeterminate contrasts, rather than as evidence for practically negligible formant differences.
12. The ROPE results differ quite sharply across representations. Acoustic F1 produced 2 non-equivalent, 8 indeterminate, and 1 insufficient contrast. Whisper produced 10 equivalent contrasts and 1 insufficient case. XLS-R sat between them, with 6 equivalent, 4 indeterminate, and 1 insufficient contrast.
13. This disagreement is meaningful rather than contradictory. Formants isolate narrow articulatory differences, while neural cosine distances pool phonetic detail with context, speaker, and model-specific information. A contrast can therefore be acoustically robust while still falling within the neural noise floor.
14. The clustering results split the phonological structure across models. Whisper layer 20 recovered vowel height best, ARI = 0.499. XLS-R layer 18 recovered front/back/central grouping best, ARI = 0.305. The acoustic representation had the clearest vowel-clustering silhouette, but not the strongest height ARI.
15. Speaker clustering revealed strong group information in the neural spaces. Whisper layer 20 separated L1/L2 perfectly at k = 2, ARI = 1.000, while XLS-R layer 18 separated gender perfectly, ARI = 1.000. This is useful, but it also means the neural embeddings should not be treated as purely phonetic.
16. For consonant/vowel clustering, XLS-R layer 18 performed best, ARI = 0.638, followed by the acoustic representation at 0.487. Whisper layer 20 was near chance, ARI = -0.024. The recurrently difficult consonants were /l/, /n/, and /ʁ/, which makes phonetic sense because sonorants and rhotics sit closer to vowels than obstruents do.

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
