"""Primary endpoint tables for the CLEAREST main study manuscript (Radiology).

Reproduces the three main-text tables that describe the cohort and the primary
(patient likelihood-of-malignancy, PLoM) endpoint:

Table 1 -- Patient demographics and CT-scan characteristics (N=240) by final
           diagnosis. The benign group is split into CTs with a benign nodule
           and nodule-negative (NN) CTs, giving a 3-group comparison
           (Malignant / Benign nodule / Nodule negative).
Table 2 -- Index-nodule characteristics (size, margins, density) by final
           diagnosis (malignant vs benign).
Table 3 -- PLoM effect size (aided - unaided AUC) by reader and index-nodule
           subgroup, from the DBM RRRC analysis.

Each table is written as a CSV to the output directory (default: ``tables/``).

Data inputs (not distributed with this repository)
--------------------------------------------------
Under ``<data-root>`` (default: ``../resources/clearest_main_study``):
  dataset/clearest_patient_data_240_main_study_max_lcp_score_gt.csv   (Tables 1, 2)
      Patient/scan-level table with ground-truth (``_gt``) fields.
  mrmc/dataset_effect_size_analysis.csv                               (Table 3)
      Per-(sub)group DBM effect sizes produced by the effect-size R analysis
      (aided/unaided AUC, effect size, RRRC p-value for each subgroup dataset).

Usage
-----
    python primary_endpoint_tables.py \
        --data-root /path/to/resources/clearest_main_study --out-dir ./tables

The data root may also be supplied via the CLEAREST_DATA_ROOT environment variable.
"""

from __future__ import annotations

import argparse
import os
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control


# --------------------------------------------------------------------------- #
# Shared preprocessing
# --------------------------------------------------------------------------- #
ATTENUATION_MAP = {"solid": 1, "part-solid": 2, "part_solid": 2, "ggo": 3}

# Manufacturer DICOM strings -> display vendor name (GE variants merged).
CT_VENDOR_MAP = {
    "GE MEDICAL SYSTEMS": "GE", "GE": "GE", "SIEMENS": "Siemens",
    "Philips": "Philips", "TOSHIBA": "Toshiba",
}


def map_size(diameter):
    """Bucket a long-axis diameter (mm) into small / medium / large."""
    if pd.isna(diameter):
        return np.nan
    if diameter < 9:
        return "small"
    if diameter < 15:
        return "medium"
    return "large"


def preprocess_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the ground-truth fields used by Tables 1 and 2."""
    df = df.copy()
    df["is_spiculated"] = df["is_spiculated_gt"].apply(lambda x: 1 if x == "yes" else 0)
    df["attenuation_gt"] = df["attenuation_gt"].map(ATTENUATION_MAP)
    df["attenuation"] = df["attenuation"].map(ATTENUATION_MAP)
    df["size_bucket_gt"] = df["diameter_long_gt"].apply(map_size)
    return df


def load_patient_gt(data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        data_root / "dataset" / "clearest_patient_data_240_main_study_max_lcp_score_gt.csv")
    return preprocess_gt(df)


# --------------------------------------------------------------------------- #
# Exact Fisher-Freeman-Halton tests (used where expected cell counts are small)
# --------------------------------------------------------------------------- #
def _log_mhg_prob(table):
    """Log multivariate-hypergeometric probability of a table given its marginals."""
    t = np.asarray(table, dtype=np.int64)
    rs, cs, n = t.sum(axis=1), t.sum(axis=0), int(t.sum())
    return (sum(lgamma(int(r) + 1) for r in rs)
            + sum(lgamma(int(c) + 1) for c in cs)
            - lgamma(n + 1)
            - sum(lgamma(int(x) + 1) for x in t.flatten()))


def fisher_freeman_halton_2xC(table):
    """Exact Fisher-Freeman-Halton two-sided p-value for a 2xC table."""
    t = np.asarray(table, dtype=np.int64)
    assert t.shape[0] == 2, f"expected 2xC table, got {t.shape}"
    cs = t.sum(axis=0)
    r0, C = int(t[0].sum()), t.shape[1]
    log_p_obs = _log_mhg_prob(t)
    log_ps = []

    def rec(j, remaining, top):
        if j == C - 1:
            if 0 <= remaining <= cs[j]:
                top_row = np.array(top + [remaining], dtype=np.int64)
                log_ps.append(_log_mhg_prob(np.array([top_row, cs - top_row], dtype=np.int64)))
            return
        for v in range(0, min(int(cs[j]), remaining) + 1):
            rec(j + 1, remaining - v, top + [v])

    rec(0, r0, [])
    log_ps = np.array(log_ps)
    mask = log_ps <= log_p_obs + 1e-12
    m = log_ps[mask].max()
    return float(np.exp(m) * np.exp(log_ps[mask] - m).sum())


def fisher_freeman_halton_3x2(table):
    """Exact Fisher-Freeman-Halton two-sided p-value for a 3x2 table."""
    t = np.asarray(table, dtype=np.int64)
    assert t.shape == (3, 2), f"expected 3x2 table, got {t.shape}"
    rs, cs = t.sum(axis=1), t.sum(axis=0)
    log_p_obs = _log_mhg_prob(t)
    log_ps = []
    for a in range(min(int(rs[0]), int(cs[0])) + 1):
        b_lo = max(0, int(cs[0]) - a - int(rs[2]))
        b_hi = min(int(rs[1]), int(cs[0]) - a)
        for b in range(b_lo, b_hi + 1):
            c = int(cs[0]) - a - b
            cand = np.array([[a, rs[0] - a], [b, rs[1] - b], [c, rs[2] - c]], dtype=np.int64)
            log_ps.append(_log_mhg_prob(cand))
    log_ps = np.array(log_ps)
    mask = log_ps <= log_p_obs + 1e-12
    m = log_ps[mask].max()
    return float(np.exp(m) * np.exp(log_ps[mask] - m).sum())


# --------------------------------------------------------------------------- #
# Table 1 -- demographics by 3-group diagnosis (Malignant / Benign nodule / NN)
# --------------------------------------------------------------------------- #
def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slice_thickness_bucket"] = pd.cut(
        df.SliceThickness, bins=[-np.inf, 1.5, np.inf], right=False,
        labels=["thin", "thick"]).astype("object")
    df["ct_vendor"] = df.Manufacturer.map(CT_VENDOR_MAP)

    mal = df[df.is_cancer == 1]
    ben_nod = df[(df.is_cancer == 0) & (df.is_nodule_gt == 1)]
    neg = df[(df.is_cancer == 0) & (df.is_nodule_gt == 0)]
    groups = [("total", df), ("mal", mal), ("ben", ben_nod), ("neg", neg)]
    outcome = groups[1:]  # the three compared groups (exclude Total)
    denoms = {k: len(g) for k, g in groups}

    def fmt_p(p):
        return "" if p == "" or pd.isna(p) else f"{p:.3f}"

    def cat_p(col, keys):
        tab = np.array([[(g[col] == k).sum() for _, g in outcome] for k in keys])
        return stats.chi2_contingency(tab)[1]

    def contains(g, term):
        return g["pulmonary_diagnosis_gt"].fillna("").str.contains(term, case=False, regex=False)

    def comorb_p(mask_fn):
        present = [int(mask_fn(g).sum()) for _, g in outcome]
        absent = [denoms[k] - p for (k, _), p in zip(outcome, present)]
        tab = np.array([present, absent])
        _, _, _, expected = stats.chi2_contingency(tab, correction=False)
        if expected.min() < 5:
            return fisher_freeman_halton_2xC(tab)
        return stats.chi2_contingency(tab, correction=False)[1]

    def npct_row(label, col, value):
        cells = [label]
        for k, g in groups:
            n = (g[col] == value).sum()
            cells.append(f"{n} ({round(100 * n / denoms[k])})")
        return cells + ["", ""]

    def med_iqr_row(label, col, p):
        cells = [label]
        for k, g in groups:
            s = g[col].dropna()
            cells.append(f"{s.median():.0f} [{s.quantile(0.25):.0f}, {s.quantile(0.75):.0f}]")
        return cells + [fmt_p(p), ""]

    def header(label, p):
        return [label, "", "", "", "", fmt_p(p), ""]

    def count_row(label, mask_fn, p="", q=""):
        cells = [label]
        for k, g in groups:
            n = int(mask_fn(g).sum())
            cells.append(f"{n} ({round(100 * n / denoms[k])})")
        return cells + [fmt_p(p), fmt_p(q)]

    rows = []
    # Age -- Kruskal-Wallis (3-group nonparametric analogue of Mann-Whitney U).
    _, p_age = stats.kruskal(*[g.age.dropna() for _, g in outcome])
    rows.append(med_iqr_row("Age (y), median [Q1, Q3]", "age", p_age))

    rows.append(header("Sex", cat_p("sex", ["F", "M"])))
    rows.append(npct_row("  Female", "sex", "F"))
    rows.append(npct_row("  Male", "sex", "M"))

    rows.append(header("Type of chest CT", cat_p("is_screening", [1, 0])))
    rows.append(npct_row("  Screening", "is_screening", 1))
    rows.append(npct_row("  Diagnostic", "is_screening", 0))

    rows.append(header("Slice thickness", cat_p("slice_thickness_bucket", ["thin", "thick"])))
    rows.append(npct_row("  0.5 to <1.5 mm", "slice_thickness_bucket", "thin"))
    rows.append(npct_row("  1.5 mm - max", "slice_thickness_bucket", "thick"))

    rows.append(header("CT-scan vendor", cat_p("ct_vendor", ["GE", "Siemens", "Philips", "Toshiba"])))
    for label, val in [("  GE", "GE"), ("  Siemens", "Siemens"),
                       ("  Philips", "Philips"), ("  Toshiba", "Toshiba")]:
        rows.append(npct_row(label, "ct_vendor", val))

    # Lung comorbidities: individual items form the Benjamini-Hochberg block;
    # the "any comorbidity" row is a separate omnibus (no q-value).
    comorbidity_items = [
        ("  Emphysema", lambda g: contains(g, "Emphysema")),
        ("  Atelectasis", lambda g: contains(g, "Atelectasis")),
        ("  Bronchiectasis", lambda g: contains(g, "Bronchiectasis")),
        ("  Fibrosis (Adj. Fibrotic)", lambda g: contains(g, "Fibrosis_(Adj._Fibrotic)")),
        ("  Pulmonary Bullae", lambda g: contains(g, "Pulmonary_Bullae")),
        ("  Consolidation", lambda g: contains(g, "Consolidation")),
        ("  Other", lambda g: g["other_pulmonary_diagnosis_gt"].notna()),
    ]
    raw_p = [comorb_p(fn) for _, fn in comorbidity_items]
    q_vals = false_discovery_control(raw_p, method="bh")
    rows.append(count_row("Lung comorbidities",
                          lambda g: g["pulmonary_diagnosis_gt"].notna(),
                          p=comorb_p(lambda g: g["pulmonary_diagnosis_gt"].notna())))
    for (label, fn), p, q in zip(comorbidity_items, raw_p, q_vals):
        rows.append(count_row(label, fn, p=p, q=q))

    cols = ["Variable", f"Total (n={denoms['total']})", f"Malignant (n={denoms['mal']})",
            f"Benign nodule (n={denoms['ben']})", f"Nodule negative (n={denoms['neg']})",
            "p-value", "q (BH)"]
    return pd.DataFrame(rows, columns=cols)


TABLE1_CAPTION = (
    "Table 1. Patient demographics and CT-scan characteristics (N=240) by final "
    "diagnosis, benign split into CTs with a benign nodule and nodule-negative CTs. "
    "Cells are n (column %). p-values are a 3-group omnibus (Malignant vs Benign "
    "nodule vs Nodule negative): Kruskal-Wallis for age; Pearson chi-square for "
    "categorical rows; comorbidities use chi-square (Fisher-Freeman-Halton exact when "
    "any expected count <5). q (BH) applies Benjamini-Hochberg across the comorbidity "
    "block; the 'Lung comorbidities' omnibus row is reported without a q-value."
)


# --------------------------------------------------------------------------- #
# Table 2 -- index-nodule characteristics by diagnosis (malignant vs benign)
# --------------------------------------------------------------------------- #
def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_n = len(df)
    mal, ben = df[df.is_cancer == 1], df[df.is_cancer == 0]
    n_mal, n_ben = len(mal), len(ben)

    def npct(series, denom, value):
        n = (series == value).sum()
        return f"{n} ({round(100 * n / denom)})"

    def chi2_p(col):
        return stats.chi2_contingency(pd.crosstab(df[col], df.is_cancer))[1]

    rows = []
    p_size = chi2_p("size_bucket_gt")
    rows.append(["Index-nodule long diameter", "", "", "", f"{p_size:.3f}"])
    for label, val in [("  Small (5-8 mm)", "small"), ("  Medium (9-14 mm)", "medium"),
                       ("  Large (>=15 mm)", "large")]:
        rows.append([label, npct(df.size_bucket_gt, total_n, val),
                     npct(mal.size_bucket_gt, n_mal, val), npct(ben.size_bucket_gt, n_ben, val), ""])

    p_spic = chi2_p("is_spiculated_gt")
    rows.append(["Index-nodule margins", "", "", "", f"{p_spic:.3f}"])
    for label, val in [("  Nonspiculated", "no"), ("  Spiculated", "yes")]:
        rows.append([label, npct(df.is_spiculated_gt, total_n, val),
                     npct(mal.is_spiculated_gt, n_mal, val), npct(ben.is_spiculated_gt, n_ben, val), ""])

    # Nodule density: 3x2 Fisher-Freeman-Halton (small expected counts in part-solid/GGO).
    p_den = fisher_freeman_halton_3x2(pd.crosstab(df.attenuation_gt, df.is_cancer).values)
    rows.append(["Nodule density", "", "", "", f"{p_den:.3f}"])
    for label, val in [("  Solid", 1), ("  Part-solid", 2), ("  GGO", 3)]:
        rows.append([label, npct(df.attenuation_gt, total_n, val),
                     npct(mal.attenuation_gt, n_mal, val), npct(ben.attenuation_gt, n_ben, val), ""])

    cols = ["Variable", f"Total (n={total_n})", f"Malignant (n={n_mal})",
            f"Benign (n={n_ben})", "p-value"]
    return pd.DataFrame(rows, columns=cols)


TABLE2_CAPTION = (
    "Table 2. Index-nodule characteristics by final diagnosis (malignant vs benign). "
    "Cells are n (column %); non-nodule cases are in the denominator but no category, "
    "so rows need not sum to 100%. Tests: Pearson chi-square for size and margins; "
    "Fisher-Freeman-Halton exact (3x2) for density (small expected counts)."
)


# --------------------------------------------------------------------------- #
# Table 3 -- PLoM effect size by reader / index-nodule subgroup
# --------------------------------------------------------------------------- #
# (label, dataset key in dataset_effect_size_analysis.csv); None key = section header.
TABLE3_SUBGROUPS = [
    ("Reader specialty", None),
    ("  Cardiothoracic", "clearest_results_specialty_yes"),
    ("  Other", "clearest_results_specialty_no"),
    ("Experience", None),
    ("  >=5 y", "clearest_results_certification_yes"),
    ("  <5 y", "clearest_results_certification_no"),
    ("Previous use of AI tools", None),
    ("  Yes", "clearest_results_use_ai_yes"),
    ("  No", "clearest_results_use_ai_no"),
    ("Confirmation bias", None),
    ("  Yes", "clearest_results_confirmation_bias_yes"),
    ("  No", "clearest_results_confirmation_bias_no"),
    ("Automation bias", None),
    ("  Yes", "clearest_results_automation_bias_yes"),
    ("  No", "clearest_results_automation_bias_no"),
    ("Aversion bias", None),
    ("  Yes", "clearest_results_aversion_bias_yes"),
    ("  No", "clearest_results_aversion_bias_no"),
    ("Trust in LCP", None),
    ("  Yes", "clearest_results_trust_in_lcp_yes"),
    ("  No", "clearest_results_trust_in_lcp_no"),
    ("Index-nodule diameter", None),
    ("  Small (5-8 mm)", "clearest_results_nodule_small"),
    ("  Medium (9-14 mm)", "clearest_results_nodule_medium"),
    ("  Large (>=15 mm)", "clearest_results_nodule_large"),
    ("Index-nodule density", None),
    ("  Solid", "clearest_results_nodule_solid"),
    ("  Part-solid", "clearest_results_nodule_part_solid"),
    ("  GGO", "clearest_results_nodule_ggo"),
    ("Index-nodule margins", None),
    ("  Nonspiculated", "clearest_results_nodule_non_spiculated"),
    ("  Spiculated", "clearest_results_nodule_spiculated"),
    ("Type of chest CT", None),
    ("  Diagnostic", "clearest_results_not_screening"),
    ("  Screening", "clearest_results_screening"),
]


def build_table3(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).set_index("dataset")

    def fmt_p(p):
        if pd.isna(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    rows = []
    for label, key in TABLE3_SUBGROUPS:
        if key is None:
            rows.append([label, "", "", "", ""])
            continue
        r = df.loc[key]
        rows.append([label, r["unaided_trt"], r["aided_trt"], r["effect_size"],
                     fmt_p(r["pvalue_rrrc"])])
    cols = ["Variable", "Avg AUC w/o CAD", "Avg AUC with CAD", "Effect size", "p-value"]
    return pd.DataFrame(rows, columns=cols)


TABLE3_CAPTION = (
    "Table 3. Average reader AUC with and without CAD, and the AI effect size "
    "(aided - unaided AUC), by prespecified reader and index-nodule subgroup. "
    "Values are mean (95% CI) in percentage points. p-values are from the DBM "
    "random-reader random-case (RRRC) analysis."
)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CLEAREST_DATA_ROOT", "../resources/clearest_main_study")),
        help="Path to the clearest_main_study data directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("tables"),
                        help="Directory to write the table CSVs into.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = load_patient_gt(args.data_root)

    tables = [
        ("table1_demographics.csv", TABLE1_CAPTION, build_table1(patients)),
        ("table2_index_nodule.csv", TABLE2_CAPTION, build_table2(patients)),
        ("table3_effect_size_by_subgroup.csv", TABLE3_CAPTION,
         build_table3(args.data_root / "mrmc" / "dataset_effect_size_analysis.csv")),
    ]
    for fname, caption, df in tables:
        print(caption)
        print(df.to_string(index=False))
        out = args.out_dir / fname
        df.to_csv(out, index=False)
        print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
