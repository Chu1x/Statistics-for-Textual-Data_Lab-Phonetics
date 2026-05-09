# Reproducible Pipeline

The project is wired as a Snakemake workflow.

## Dry Run

Use a writable Snakemake cache directory on macOS:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake -n --cores 1
```

If all files are present, Snakemake should report that nothing needs to be done.

## Run Everything

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake --cores 1
```

The main workflow targets are:

```text
results/report.md
results/tables/acoustic_missingness_by_phoneme_group.csv
```

## Configuration

Workflow parameters are stored in:

```text
config/config.yaml
```

Important settings:

- `runtime.local_files_only: true` uses already-cached Hugging Face models.
- Set `runtime.local_files_only: false` if the models need to be downloaded.
- Whisper layers are configured under `layers.whisper`.
- XLS-R layers are configured under `layers.xlsr`.
- Bootstrap/permutation counts are configured under `runtime`.

## Main Rules

- `parse_corpus`
- `extract_acoustics`
- `extract_neural_whisper`
- `extract_neural_xlsr`
- `normalise`
- `analyse_descriptive`
- `analyse_statistical`
- `analyse_trajectory`
- `analyse_quality`
- `analyse_sensitivity`
- `analyse_mixed_models`
- `analyse_rope`
- `analyse_clustering`
- `write_report`

## Outputs

Core data outputs are written to `data/`.

Tables and figures are written to:

```text
results/tables/
results/figures/
```

Key submitted table outputs include:

- `results/tables/acoustic_missingness_by_phoneme_group.csv` - acoustic missingness by phoneme, L1/L2 group and gender.
- `results/tables/acoustic_vowel_descriptives.csv` - vowel-level acoustic descriptives.
- `results/tables/acoustic_l1_l2_tests.csv` - acoustic L1/L2 tests with FDR correction.
- `results/tables/neural_l1_l2_permutation_whisper_layer20.csv` and `results/tables/neural_l1_l2_permutation_xlsr_layer18.csv` - neural permutation tests.
- `results/tables/mixed_model_comparisons.csv` and `results/tables/mixed_model_representation_r2_summary.csv` - mixed-model comparisons and R2 summaries.
- `results/tables/rope_summary.csv` - combined ROPE summary.
- `results/tables/clustering_vowel_ari.csv`, `results/tables/clustering_consonant_vowel_ari.csv`, and `results/tables/clustering_speaker_ari.csv` - clustering metrics.

The final Markdown report is:

```text
results/report.md
```

The final PDF/Word files are exported from the Markdown source with Pandoc, using `markdown-implicit_figures` so that figure titles in the report are not converted into duplicate captions.
