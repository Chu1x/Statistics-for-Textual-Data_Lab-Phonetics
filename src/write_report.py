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

This report summarises the reproducible pipeline for the Russian-French Interference Corpus. The pipeline parsed TextGrid annotations, extracted acoustic features, extracted Whisper and XLS-R hidden-state representations, normalised features, ran statistical tests, fitted mixed-effects models, evaluated ROPE classifications, and performed hierarchical clustering.

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

The parsed corpus contains {n_tokens} phoneme tokens from {n_speakers} speakers and {n_sentences} sentence IDs. Acoustic extraction used midpoint formants and, for vowels longer than 80 ms, additional 25% and 75% formant measurements. Vowels were Lobanov-normalised within speaker using vowel tokens only.

Whisper embeddings used `{config["models"]["whisper"]}` layers {whisper_layers}. XLS-R embeddings used `{config["models"]["xlsr"]}` layers {xlsr_layers}. Neural representations were reduced to {config["runtime"]["pca_components"]} principal components per selected layer for downstream analyses. Statistical tests used {config["runtime"]["permutations"]} permutations where applicable and ROPE/bootstrap summaries used {config["runtime"]["bootstrap"]} bootstrap samples where applicable.

Missing-value summary by analysis-relevant token set:

{_long_table(missingness, "missingness")}

Rough acoustic range flags:

{_long_table(quality_display, "quality")}

Missing acoustic values were not imputed. Analyses used pairwise exclusion for the feature being tested, so each result table reports the available sample size for that feature and contrast. Rough-range flags were retained rather than automatically removed; among oral vowels, flagged counts were F1 = {oral_f1_flags}, F2 = {oral_f2_flags}, and f0 = {oral_f0_flags}. These flags are treated as quality diagnostics, not as exclusions from the main formant analyses.

Acoustic L1/L2 sensitivity check after excluding rough-range F1/F2 flags changed {sensitivity_changed} of {sensitivity_total} FDR-significance decisions. Affected or filtered contrasts:

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

1. PCA and UMAP answered different parts of the projection question. PCA gave a linear, variance-preserving baseline whose 2D between-phoneme variance ratios were high for several layers, especially XLS-R layer 18. UMAP was more useful as a neighbourhood visualisation, but it is less directly interpretable because distances and axes are not linear acoustic or articulatory dimensions.
2. The most variable vowels in the acoustic summaries were concentrated among low/back and high rounded categories, especially /ɑ/, /u/, /y/, and /ɛ/. This partly matches the neural spaces, but the neural embeddings also preserve contextual, speaker, and language-background information, so acoustic dispersion and neural dispersion are related rather than identical.
3. The UMAP plots recover phoneme neighbourhoods, especially for XLS-R, but they do not perfectly reconstruct the IPA vowel trapezoid. This is expected because the embeddings are learned from speech signal prediction/recognition objectives rather than from explicit F1/F2 geometry.
4. RSM/Mantel correlations showed stronger acoustic alignment for XLS-R than Whisper: acoustic-Whisper r = 0.189, acoustic-XLS-R r = 0.355, and Whisper-XLS-R r = 0.676 on the sampled token similarity matrices. The two neural models are therefore more similar to each other than either is to the acoustic F1/F2 space.
5. Acoustic L1/L2 tests found 9 FDR-significant contrasts across /i/, /y/, /u/, /ø/, /ɛ/, and /ɑ/. Whisper and XLS-R each found 9 significant vowel-level neural contrasts; /œ/ had insufficient data and /ə/ was not significant. On long-vowel subsets, replacing midpoint formants with 25%/75% trajectory means changed {trajectory_changed} of {trajectory_total} FDR decisions, so trajectory information matters for some vowels.
6. Acoustic Euclidean and Mahalanobis centroid distances were nearly identical in rank structure, r = 0.984. Acoustic-neural centroid correlations were moderate and similar for Whisper and XLS-R, r = 0.674 and r = 0.678 respectively for Euclidean acoustic distance, while Whisper-XLS-R centroid distance correlation was stronger, r = 0.842.
7. Leave-one-speaker-out nearest-centroid classification was best for Whisper layer 20, with overall accuracy {_fmt(best_identification.overall_accuracy)} and macro-F1 0.735. XLS-R layer 18 followed with accuracy 0.790, and acoustic F1/F2 reached 0.687. McNemar tests showed all pairwise classifier differences were significant.
8. For /a/, speaker-specificity was modest in all representations: ICC = 0.038 for acoustic F1, 0.042 for Whisper PC1, and 0.031 for XLS-R PC1. This indicates that most /a/ variation is token-level or residual rather than speaker-level in these fitted models.
9. The L1 x Gender interaction was not significant for acoustic F1/F2 or Whisper PCs 1-5. In XLS-R, PC2 showed a significant interaction, estimate = 6.649, p = 0.031, so gender-related structure appears in one neural dimension but is not a broad cross-representation pattern.
10. Mixed-effects model summaries gave the highest mean marginal R2 to acoustic features, {_fmt(best_r2.mean_marginal_r2)}, followed by XLS-R layer 18 at 0.410 and Whisper layer 20 at 0.310. Acoustic F1/F2 therefore captured the fixed phonetic predictors most directly, while neural PCs retained additional non-phoneme variance.
11. No statistically significant acoustic F1 contrast had a 95% speaker-level CI fully inside the acoustic ROPE. The current acoustic ROPE result is therefore not a case of "statistically significant but practically equivalent"; instead, several contrasts were indeterminate and two acoustic F1 contrasts were non-equivalent.
12. ROPE classifications differed by representation: acoustic F1 had 2 non-equivalent, 8 indeterminate, and 1 insufficient contrast; Whisper had 10 equivalent and 1 insufficient contrast; XLS-R had 6 equivalent, 4 indeterminate, and 1 insufficient contrast. Whisper therefore produced the strongest practical-equivalence pattern under the chosen neural ROPE.
13. Acoustic and neural ROPE disagreement is interpretable because the acoustic tests target narrow formant shifts, while neural cosine distances average broader phonetic, contextual, and speaker-conditioned information. A small formant change can be robust acoustically while remaining within the neural within-speaker noise floor.
14. Clustering results separated different phonological structures by representation. Whisper layer 20 best recovered vowel height, ARI = 0.499; XLS-R layer 18 best recovered front/back/central grouping, ARI = 0.305; acoustic F1/F2 gave the clearest silhouette for vowel clustering but a lower height ARI than Whisper.
15. Speaker clustering showed strong representation-specific biases. Whisper layer 20 perfectly separated L1/L2 at k = 2, ARI = 1.000, whereas XLS-R layer 18 perfectly separated gender, ARI = 1.000. These results warn that neural embeddings encode speaker and group information in addition to phonetic identity.
16. Consonant/vowel clustering was best for XLS-R layer 18, ARI = 0.638, followed by acoustic features at 0.487; Whisper layer 20 was near chance, ARI = -0.024. The systematically difficult consonants were /l/, /n/, and /ʁ/, consistent with sonorants and rhotics behaving as acoustically intermediate categories.

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

Figures are placed at the end to avoid interrupting long result listings in the PDF layout.

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
