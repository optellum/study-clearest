# CLEAREST main study — analysis scripts

Analysis code supporting the CLEAREST main-study manuscript (submitted to *Radiology*).
CLEAREST is a retrospective, fully-crossed multi-reader multi-case (MRMC) study evaluating an
integrated CT-based computer-aided detection and diagnosis (CADe/CADx) tool for pulmonary-nodule
assessment: 15 radiologists read 240 chest-CT examinations both **unaided** and **aided** by the AI.

These scripts reproduce the statistics, figures, and tables reported in the manuscript. They are
released so reviewers and readers can inspect exactly how each result was computed.

The study data are **not** included in this repository. Each script reads from a local data tree, 
whose location you supply at run time (see below).

## Requirements

- **Python 3.11** with the packages pinned in `pyproject.toml` / `uv.lock` (also exported to
  `requirements.txt`): pandas, numpy, scikit-learn, scipy, matplotlib, pydicom.
  Dependencies are managed with [uv](https://docs.astral.sh/uv/).
- **R** (≥ 4.1) with `RJafroc`, `pROC`, `dplyr`, `lme4`, and `lmerTest`.

## Setup and running the Python scripts (uv)

```bash
# create the environment from the lockfile
uv sync

# run a script, pointing --data-root at your local clearest_main_study directory
uv run python primary_endpoint_figures.py   --data-root /path/to/clearest_main_study --out-dir figures
uv run python secondary_endpoint_figures.py --data-root /path/to/clearest_main_study --out-dir figures
uv run python primary_endpoint_tables.py    --data-root /path/to/clearest_main_study --out-dir tables
uv run python secondary_endpoint_tables.py  --data-root /path/to/clearest_main_study --out-dir tables
```

The data root may also be set once via the `CLEAREST_DATA_ROOT` environment variable instead of
`--data-root`. Figure scripts write PNGs to `--out-dir` (default `figures/`); table scripts write
CSVs to `--out-dir` (default `tables/`). Pass `--show` to a figure script to display the figures
interactively as well as saving them.

## Running the R scripts

```bash
Rscript effect_size_cohort.R
```

Each R script reads its inputs (iMRMC / CSV files) from the data tree and prints its results
(some also write a summary CSV back into the data tree).

## Scripts and the manuscript outputs they produce

| Script | Manuscript output |
| --- | --- |
| `primary_endpoint_figures.py` | Figure 4 (average reader ROC), Figure 5 (AUC by sub-specialty), Figure E1 (per-reader effect size) |
| `secondary_endpoint_figures.py` | Figure 6 (detection FROC operating point), Figure E2 (Bland–Altman of reader vs ground-truth diameter) |
| `primary_endpoint_tables.py` | Table 1 (demographics), Table 2 (index-nodule characteristics), Table 3 (effect size by subgroup) |
| `secondary_endpoint_tables.py` | Table 4 (inter-reader agreement), Table E1 (ground-truth counts), Table E4 (detection performance) |
| `effect_size_cohort.R` | Primary endpoint: cohort aided-vs-unaided AUC effect size (DBM, RRRC) |
| `effect_size_subgroup.R` | Effect size by reader human-factors / AI-attitude subgroup, with a yes-vs-no z-test |
| `effect_size_subgroup_interaction.R` | Treatment × subgroup interaction (effect-modification) tests |
| `effect_size_individual_reader.R` | Per-reader DeLong AUC comparison (aided vs unaided; feeds Figure E1) |
| `read_time.R` | Secondary endpoint: change in reading time (linear mixed model) |

Each script's header documents its statistical method and the expected input data format.

Not script-generated: Figure 1 (example CADe/CADx report), Figure 2 (CONSORT flow diagram), and
Figure 3 (reading-platform workflow) are illustrative; the case-selection step is not part of this
repository.

## Statistical methods

- **DBM** — Dorfman–Berbaum–Metz (Hillis-modified) MRMC ROC analysis (`RJafroc`), with the Wilcoxon
  (empirical) AUC figure of merit and a random-reader random-case (RRRC) generalization.
- **DeLong test** — comparison of two correlated (paired) ROC AUCs (`pROC`).
- **Fleiss' κ** — multi-rater categorical agreement, with percentile-bootstrap confidence intervals.
- **Linear mixed model** — log reading time with crossed reader and case random effects (`lme4` /
  `lmerTest`).

## Notes

- Generated outputs (figures, tables, and any data-derived CSVs) are not tracked in git.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
