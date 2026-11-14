# CLEAREST main study — analysis scripts

Analysis code supporting the CLEAREST main-study manuscript (*Radiology*, RAD-26-2244).
CLEAREST is a retrospective, fully crossed multi-reader multi-case (MRMC) study evaluating an
integrated CT-based computer-aided detection and diagnosis (CADe/CADx) tool for pulmonary-nodule
assessment: 15 radiologists read 240 chest CT examinations both **unaided** and **aided** by the AI.

These scripts reproduce the statistics, tables and figures reported in the manuscript and its
supplemental material. They are released so reviewers and readers can inspect exactly how each
result was computed.

The study data are **not** included in this repository. Each script reads from a local data
tree (`clearest_main_study/`) whose location you supply at run time (see below).

## Requirements

- **Python 3.11** with the packages pinned in `pyproject.toml` / `uv.lock` (also exported to
  `requirements.txt`): pandas, numpy, scikit-learn, scipy, matplotlib.
  Dependencies are managed with [uv](https://docs.astral.sh/uv/).
- **R** (≥ 4.4) with `RJafroc` (2.1.2), `pROC`, `dplyr`, `lme4` and `lmerTest`.

## Running the scripts

**Python** scripts take the data root as an argument (or via the `CLEAREST_DATA_ROOT`
environment variable) and write to `--out-dir`:

```bash
uv sync
uv run python primary_endpoint_tables.py --data-root /path/to/clearest_main_study --out-dir tables
```

Figure scripts write PNGs (default `figures/`; `--show` also displays them); table scripts write
CSVs (default `tables/`).

**R** scripts read and write paths relative to `resources/clearest_main_study/`, so run them from
the directory that contains `resources/`:

```bash
cd /path/to/parent-of-resources
Rscript /path/to/study-clearest/effect_size_cohort.R
```

## Run order

Some Python scripts read CSVs written by the R scripts. To regenerate everything:

1. `Rscript effect_size_cohort.R` → `mrmc/cohort_auc_by_treatment.csv` (Figure 3)
2. `Rscript effect_size_individual_reader.R` → `mrmc/plom_pvalue.csv` (Figure E1 asterisks)
3. `Rscript effect_size_subgroup_dbm.R` → `mrmc/dataset_effect_size_analysis.csv` (Table 3)
4. `Rscript effect_size_subgroup_interaction.R` → `mrmc/dataset_subgroup_interaction_analysis.csv` (Table 3)
5. `Rscript reader_group_contrasts.R`, `Rscript device_vs_readers.R` (Tables E6, E7)
6. The Python table and figure scripts, in any order.

`read_time.R` and `effect_size_subgroup.R` are independent.

## Manuscript outputs and the scripts that produce them

### Main text

| Output | Script | Notes |
| --- | --- | --- |
| Table 1 — patient demographics and CT characteristics | `primary_endpoint_tables.py` | `table1_demographics.csv` |
| Table 2 — index-nodule characteristics | `primary_endpoint_tables.py` | `table2_index_nodule.csv`; pre-specified size groups <9 / 9 to <15 / ≥15 mm (long axis, unrounded) |
| Table 3 — PLoM effect size by reader and index-nodule subgroup | `effect_size_subgroup_dbm.R` → `effect_size_subgroup_interaction.R` → `primary_endpoint_tables.py` | DBM RRRC per subgroup; P for interaction with Benjamini–Hochberg adjustment across 11 pre-specified factors; experience = years of practice |
| Table 4 — inter-reader agreement (Fleiss' κ) | `secondary_endpoint_tables.py` | `table4_agreement.csv`; paired patient-cluster bootstrap |
| Figure 1 — CONSORT flow diagram | — | illustrative, not script-generated |
| Figure 2 — study platform and device report | — | screenshot, not script-generated |
| Figure 3 — average reader ROC curves | `primary_endpoint_figures.py` | reads `effect_size_cohort.R` output |
| Figure 4 — AUC by sub-specialty | `primary_endpoint_figures.py` | |
| Figure 5 — detection operating points | `secondary_endpoint_figures.py` | |

Primary endpoint (cohort effect size, Results text): `effect_size_cohort.R`. Reading time
(secondary endpoint): `read_time.R`.

### Supplemental material

| Output | Script | Notes |
| --- | --- | --- |
| Table E1 — expert-adjudication reference nodules | `secondary_endpoint_tables.py` | `tableE1_ground_truth_counts.csv` |
| Table E2 — reader characteristics | `primary_endpoint_tables.py` | `tableE2_reader_characteristics.csv` |
| Table E3 — technology acceptance survey | — | survey summary; not script-generated in this repository |
| Table E4 — detection sensitivity by size | `secondary_endpoint_tables.py` | `tableE4_detection_performance.csv`, `tableE4_sensitivity_by_size.csv` (small = 5 to <9 mm) |
| Table E5 — sensitivity and specificity at the ACCP 5% and 65% thresholds | `secondary_endpoint_tables.py` | `tableE5_threshold_sensitivity_specificity.csv` |
| Table E6 — PLoM effect size by sub-specialty × experience | `reader_group_contrasts.R` | `tableE6_reader_groups.csv`; comparisons in the table note (Holm) and omnibus ANOVA: `tableE6_comparisons.csv` |
| Table E7 — readers vs device alone | `device_vs_readers.R` | `device_vs_readers.csv`; reader AUCs from `effect_size_cohort.R` (run first) |
| Figure E1 — per-reader effect size | `primary_endpoint_figures.py` | asterisks from `effect_size_individual_reader.R` |
| Figure E2 — Bland–Altman of diameter vs reference | `secondary_endpoint_figures.py` | |

### Analyses reported in the response to reviewers

| Script | Analysis |
| --- | --- |
| `false_positive_marks.py` | Reader false-positive marks with and without CAD (R1-4) |
| `aided_measurement_acceptance.py` | Aided measurements accepted unchanged from the device (R1-5) |
| `secondary_endpoint_tables.py` | `recommendation_change_nodule_cts.csv`: net change in clinical recommendation with CAD |
| `effect_size_subgroup.R` | Effect size by reader attitude subgroup with a yes-vs-no z-test (`ux/ux_analysis_dbm.csv`) |

Each script's header documents its statistical method and expected input format.

## Inputs produced upstream

The iMRMC files (`mrmc/all/*_patient_lom.imrmc` and the matching `.csv` files, including the
subgroup files) and the detection-matching tables (`detection_rate/matched_*.csv`) are prepared
from the reader exports by the study's data-preparation pipeline, which is not part of this
repository. Case selection is also not included.

## Statistical methods

- **DBM** — Dorfman–Berbaum–Metz (Hillis-modified) MRMC ROC analysis (`RJafroc`), Wilcoxon
  (empirical) AUC, random-reader random-case (RRRC) generalization.
- **DeLong test** — comparison of two correlated (paired) ROC AUCs (`pROC`).
- **Subgroup interaction** — one-way ANOVA on DBM pseudovalue differences averaged to the
  factor's sampling unit; Benjamini–Hochberg adjustment.
- **Fleiss' κ** — multi-rater categorical agreement, paired patient-cluster percentile bootstrap.
- **Linear mixed model** — log reading time (`lme4` / `lmerTest`).

## Notes

- Generated outputs (figures, tables and any data-derived CSVs) are not tracked in git.
- Results are findings of this reader study, not device performance claims.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
