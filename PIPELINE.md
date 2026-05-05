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

The workflow target is `results/report.md`.

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

The final Markdown report is:

```text
results/report.md
```
