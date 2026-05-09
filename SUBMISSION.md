# Submission Checklist

## Primary Files to Submit

- `results/report_final.pdf` - final report for grading.
- `results/report_final.docx` - editable copy, useful if the platform accepts Word files.
- `results/report.md` - source version of the report.
- `PIPELINE.md` - reproducibility instructions.
- `Snakefile` - Snakemake workflow.
- `config/config.yaml` - workflow configuration.
- `src/` - analysis and extraction scripts.
- `results/tables/` - generated result tables.
- `results/figures/` - generated figures used by the report.

## Usually Do Not Submit

- `.venv/` - local Python environment, too large and machine-specific.
- `.snakemake/` - workflow cache/metadata.
- `.DS_Store`, `results/.Rhistory`, editor backup files.
- The full `ru-fr_interference/` corpus unless the instructor explicitly requests data upload.
- Large neural arrays in `data/*.npz` unless the instructor asks for intermediate features.

## Reproducibility Commands

Dry-run the workflow:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake -n --cores 1 results/report.md results/tables/acoustic_missingness_by_phoneme_group.csv
```

Run the workflow:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake --cores 1 results/report.md results/tables/acoustic_missingness_by_phoneme_group.csv
```

Export final PDF and Word report:

```bash
.venv/bin/python src/write_report.py --output results/report.md
pandoc -f markdown-implicit_figures results/report.md --resource-path=results:. -o results/report_final.docx
printf '\\usepackage{needspace}\n' > /private/tmp/report_header.tex
pandoc -f markdown-implicit_figures results/report.md --resource-path=results:. -o results/report_final.pdf --pdf-engine=xelatex -V geometry:margin=0.8in -V mainfont='Times New Roman' -H /private/tmp/report_header.tex
```

## Final Sanity Checks

- Open `results/report_final.pdf` and inspect the Figures section manually.
- Confirm the 16 answers are present.
- Confirm `results/tables/acoustic_missingness_by_phoneme_group.csv` is present.
- Confirm tables do not overlap in the PDF.
- Confirm the Figures heading starts on its own page.
