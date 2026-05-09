# Acoustic and Neural Representations in a Phonetically Aligned Corpus

This repository contains a reproducible analysis pipeline for a phonetics project on the Russian-French Interference Corpus. The workflow parses phoneme-aligned TextGrid annotations, extracts acoustic features, extracts Whisper and XLS-R neural representations, normalises features, runs statistical analyses, and writes the final report.

## Final Report

- Final PDF: `results/report_final.pdf`
- Editable Word copy: `results/report_final.docx`
- Markdown source: `results/report.md`

## Reproducibility

The workflow is managed with Snakemake:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake -n --cores 1 results/report.md results/tables/acoustic_missingness_by_phoneme_group.csv
```

To run the full workflow:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake --cores 1 results/report.md results/tables/acoustic_missingness_by_phoneme_group.csv
```

Configuration is stored in `config/config.yaml`.

## Repository Contents

- `src/` - parsing, extraction, normalisation, analysis, and report-writing scripts.
- `Snakefile` - workflow definition.
- `config/` - model, layer, and runtime settings.
- `results/tables/` - generated result tables.
- `results/figures/` - generated figures.
- `PIPELINE.md` - workflow notes.
- `SUBMISSION.md` - submission checklist.

One important quality-control table is `results/tables/acoustic_missingness_by_phoneme_group.csv`, which reports missing acoustic values by phoneme, L1/L2 group, and gender.

The raw corpus, local virtual environment, Snakemake metadata, and large neural `.npz` arrays are intentionally excluded from version control.
