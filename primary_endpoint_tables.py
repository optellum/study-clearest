"""Primary endpoint tables for the CLEAREST main study manuscript (Radiology).

Reproduces the three main-text tables that describe the cohort and the primary
(patient likelihood-of-malignancy, PLoM) endpoint, and supplemental Table E2:

Table 1 -- Patient demographics and CT-scan characteristics (N=240) by final
           diagnosis. The benign group is split into CTs with a benign nodule
           and nodule-negative (NN) CTs, giving a 3-group comparison
           (Malignant / Benign nodule / Nodule negative).
Table 2 -- Index-nodule characteristics (size, margins, density) by final
           diagnosis (malignant vs benign); size groups <9, 9 to <15, >=15 mm
           (long-axis diameter).
Table 3 -- PLoM effect size (aided - unaided AUC) by reader and index-nodule
           subgroup, from the DBM RRRC analysis, with a treatment x subgroup
           interaction P value per factor.
Table E2 -- Reader characteristics (n = 15): sub-specialty, years of
           independent practice, region (UK / EU) and academic practice.

Each table is written as a CSV to the output directory (default: ``tables/``).

Data inputs (not distributed with this repository)
--------------------------------------------------
Under ``<data-root>`` (default: ``../resources/clearest_main_study``):
  dataset/clearest_patient_data_240_main_study_max_lcp_score_gt.csv   (Tables 1, 2)
      Patient/scan-level table with ground-truth (``_gt``) fields.
  mrmc/dataset_effect_size_analysis.csv                               (Table 3)
      Per-(sub)group DBM effect sizes produced by the effect-size R analysis
      (aided/unaided AUC, effect size, RRRC p-value for each subgroup dataset).
  mrmc/dataset_subgroup_interaction_analysis.csv                      (Table 3)
      Treatment x subgroup interaction P value per factor, written by
      effect_size_subgroup_interaction.R. Optional: if absent, the interaction
      columns are left blank.
  dataset/participant_demographic.csv                                 (Table E2)
      One row per reader: participant_id, sub_specialty, exp_y (years of
      practice), practice (institution, "<name>, <country>"), academic (Yes/No).
      The region (UK / EU) is taken from the institution's country; the
      institution itself is not output.

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
    """Bucket an unrounded long-axis diameter (mm) into the prespecified
    small (<9) / medium (9 to <15) / large (>=15 mm) groups."""
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
    # Prespecified size groups on the unrounded long-axis diameter.
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
        return cells + [""]

    def med_iqr_row(label, col, p):
        cells = [label]
        for k, g in groups:
            s = g[col].dropna()
            cells.append(f"{s.median():.0f} [{s.quantile(0.25):.0f}, {s.quantile(0.75):.0f}]")
        return cells + [fmt_p(p)]

    def header(label, p):
        return [label, "", "", "", "", fmt_p(p)]

    def count_row(label, mask_fn, p_text):
        cells = [label]
        for k, g in groups:
            n = int(mask_fn(g).sum())
            cells.append(f"{n} ({round(100 * n / denoms[k])})")
        return cells + [p_text]

    rows = []
    # Age -- Kruskal-Wallis (3-group nonparametric analogue of Mann-Whitney U).
    _, p_age = stats.kruskal(*[g.age.dropna() for _, g in outcome])
    rows.append(med_iqr_row("Age (y) *", "age", p_age))

    rows.append(header("Sex", cat_p("sex", ["F", "M"])))
    rows.append(npct_row("  Female", "sex", "F"))
    rows.append(npct_row("  Male", "sex", "M"))

    rows.append(header("Type of CT", cat_p("is_screening", [1, 0])))
    rows.append(npct_row("  Screening", "is_screening", 1))
    rows.append(npct_row("  Diagnostic", "is_screening", 0))

    rows.append(header("Section thickness", cat_p("slice_thickness_bucket", ["thin", "thick"])))
    rows.append(npct_row("  0.5 to <1.5 mm", "slice_thickness_bucket", "thin"))
    thick_max = df.SliceThickness.max()
    rows.append(npct_row(f"  1.5 - {thick_max:g} mm", "slice_thickness_bucket", "thick"))

    rows.append(header("CT-vendors", cat_p("ct_vendor", ["GE", "Siemens", "Philips", "Toshiba"])))
    for label, val in [("  GE Healthcare", "GE"), ("  Siemens", "Siemens"),
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
    # "†": any comorbidity vs none, not part of the BH block.
    p_any = comorb_p(lambda g: g["pulmonary_diagnosis_gt"].notna())
    rows.append(count_row("Lung Comorbidities", lambda g: g["pulmonary_diagnosis_gt"].notna(),
                          fmt_p(p_any) + "†"))
    for (label, fn), p, q in zip(comorbidity_items, raw_p, q_vals):
        rows.append(count_row(label, fn, f"{fmt_p(p)} / {fmt_p(q)}"))

    cols = ["Variable", f"Total (n={denoms['total']})", f"Cancer (n={denoms['mal']})",
            f"Benign (n={denoms['ben']})", f"NN (n={denoms['neg']})",
            "P value (unadjusted / adjusted)"]
    return pd.DataFrame(rows, columns=cols)


TABLE1_CAPTION = (
    "Table 1. Patient demographics, CT characteristics, and lung comorbidities by final"
    " diagnosis (cancer, benign nodule, or nodule-negative) (N = 240).\n"
    "Note – Unless otherwise indicated, data are numbers of examinations, with column "
    "percentages in parentheses. P values compare the cancer, benign-nodule, and "
    "nodule-negative groups. For the individual lung comorbidities, unadjusted / "
    "adjusted P values are shown, adjusted with the Benjamini-Hochberg "
    "false-discovery-rate procedure across the seven comorbidities; all other P values "
    "are unadjusted. Statistical tests: age, Kruskal-Wallis test; sex, type of CT, "
    "section thickness, and CT vendor, Pearson chi-square test of independence; each "
    "comorbidity, treated as an independent binary (present/absent) variable, Pearson "
    "chi-square test, or Fisher-Freeman-Halton exact test when any expected cell count "
    "was less than 5. NN = nodule-negative. * Data are medians, with first and third "
    "quartiles in brackets. † Any lung comorbidity versus none, tested as for the "
    "individual comorbidities; not included in the Benjamini-Hochberg adjustment."
)


# --------------------------------------------------------------------------- #
# Table 2 -- index-nodule characteristics by diagnosis (malignant vs benign)
# --------------------------------------------------------------------------- #
def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Restrict to index nodules (non-nodule cases have no GT diameter): column % use 188 / 95 / 93.
    df = df[df.diameter_long_gt.notna()]
    assert len(df) == 188
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
    rows.append(["Diameter", "", "", "", fmt_p(p_size)])
    for label, val in [("  Small (<9 mm)", "small"), ("  Medium (9 to <15 mm)", "medium"),
                       ("  Large (≥15 mm)", "large")]:
        rows.append([label, npct(df.size_bucket_gt, total_n, val),
                     npct(mal.size_bucket_gt, n_mal, val), npct(ben.size_bucket_gt, n_ben, val), ""])

    p_spic = chi2_p("is_spiculated_gt")
    rows.append(["Margins", "", "", "", fmt_p(p_spic)])
    for label, val in [("  Non-spiculated", "no"), ("  Spiculated", "yes")]:
        rows.append([label, npct(df.is_spiculated_gt, total_n, val),
                     npct(mal.is_spiculated_gt, n_mal, val), npct(ben.is_spiculated_gt, n_ben, val), ""])

    # Nodule density: 3x2 Fisher-Freeman-Halton (small expected counts in part-solid/GGO).
    p_den = fisher_freeman_halton_3x2(pd.crosstab(df.attenuation_gt, df.is_cancer).values)
    rows.append(["Attenuation", "", "", "", fmt_p(p_den)])
    for label, val in [("  Solid", 1), ("  Part-solid", 2), ("  Ground Glass", 3)]:
        rows.append([label, npct(df.attenuation_gt, total_n, val),
                     npct(mal.attenuation_gt, n_mal, val), npct(ben.attenuation_gt, n_ben, val), ""])

    cols = ["Variable", f"Total (n={total_n})", f"Cancer (n={n_mal})",
            f"Benign (n={n_ben})", "P value"]
    return pd.DataFrame(rows, columns=cols)


TABLE2_CAPTION = (
    "Table 2. Index-nodule characteristics (N = 188).\n"
    "Note – Data reported as counts (column-wise %). Statistical tests: Index-nodule "
    "long-diameter category and index-nodule margins: Pearson chi-square test of "
    "independence on the (malignant vs. benign); Nodule attenuation: "
    "Fisher-Freeman-Halton exact tests. Percentages are column-wise within the 188 "
    "index nodules."
)


# --------------------------------------------------------------------------- #
# Table 3 -- PLoM effect size by reader / index-nodule subgroup
# --------------------------------------------------------------------------- #
# (label, dataset key in dataset_effect_size_analysis.csv); None key = section header.
TABLE3_SUBGROUPS = [
    ("Reader specialty", None),
    ("  Cardiothoracic", "clearest_results_specialty_yes"),
    ("  Other", "clearest_results_specialty_no"),
    ("Reader Experience", None),
    ("  ≥5 y", "clearest_results_experience_yes"),
    ("  <5 y", "clearest_results_experience_no"),
    ("Prior use of AI tools", None),
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
    ("Type of CT-scan", None),
    ("  Diagnostic", "clearest_results_not_screening"),
    ("  Screening", "clearest_results_screening"),
    ("Index-nodule diameter", None),
    ("  Small (<9 mm)", "clearest_results_nodule_small"),
    ("  Medium (9 to <15 mm)", "clearest_results_nodule_medium"),
    ("  Large (≥15 mm)", "clearest_results_nodule_large"),
    ("Index-nodule attenuation", None),
    ("  Solid", "clearest_results_nodule_solid"),
    ("  Part-solid", "clearest_results_nodule_part_solid"),
    ("  Ground glass", "clearest_results_nodule_ggo"),
    ("Index-nodule margins", None),
    ("  Nonspiculated", "clearest_results_nodule_non_spiculated"),
    ("  Spiculated", "clearest_results_nodule_spiculated"),
]


# Section header -> comparison name in dataset_subgroup_interaction_analysis.csv.
# Experience = years of practice (exp_y >= 5; 5 vs 10 readers). The 11 pre-specified
# factors form the Benjamini-Hochberg family.
TABLE3_INTERACTION = {
    "Reader specialty": "specialty",
    "Reader Experience": "experience",
    "Prior use of AI tools": "use_ai",
    "Confirmation bias": "confirmation_bias",
    "Automation bias": "automation_bias",
    "Aversion bias": "aversion_bias",
    "Trust in LCP": "trust_in_lcp",
    "Index-nodule diameter": "nodule_size",
    "Index-nodule attenuation": "attenuation",
    "Index-nodule margins": "spiculation",
    "Type of CT-scan": "screening",
}


def fmt_p(p) -> str:
    """P value in Radiology style: no leading zero; two digits, three if < .01 or
    if two digits would read .05; exact unless < .001; > .99 at the top."""
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "<.001"
    if p > 0.99:
        return ">.99"
    out = f"{p:.3f}" if p < 0.01 or f"{p:.2f}" == "0.05" else f"{p:.2f}"
    return out.lstrip("0")


def table3_interaction_p(csv_path: Path) -> tuple[dict, dict]:
    """Unadjusted and BH-adjusted interaction P values keyed by comparison name."""
    if not csv_path.exists():
        return {}, {}
    df = pd.read_csv(csv_path)
    df = df[df.endpoint == "patient_lom"].set_index("comparison")
    pcol = "p_interaction_raw" if "p_interaction_raw" in df.columns else "p_interaction"
    raw = {name: df.loc[name, pcol]
           for name in TABLE3_INTERACTION.values() if name in df.index}
    family = list(raw)
    assert len(family) == 11, family
    adj = false_discovery_control([raw[n] for n in family], method="bh")
    return raw, dict(zip(family, adj))


def build_table3(csv_path: Path, interaction_csv: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path).set_index("dataset")
    raw_p, adj_p = table3_interaction_p(interaction_csv) if interaction_csv else ({}, {})

    def fmt_auc(est, lo, hi, cap=True):
        """AUC-scale value and 95% CI to two decimals; AUC CIs truncated to [0, 1]."""
        if cap:
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        # Two decimals, no negative zero, Unicode minus sign as in the manuscript.
        f2 = lambda x: f"{x:.2f}".replace("-0.00", "0.00").replace("-", "−")
        return f"{f2(est)} ({f2(lo)}, {f2(hi)})"

    rows = []
    for label, key in TABLE3_SUBGROUPS:
        if key is None:
            name = TABLE3_INTERACTION.get(label)
            p_int = (f"{fmt_p(raw_p[name])} / {fmt_p(adj_p[name])}" if name in raw_p else "")
            rows.append([label, "", "", "", "", p_int])
            continue
        r = df.loc[key]
        rows.append([label,
                     fmt_auc(r["unaided_raw"], r["unaided_lo_raw"], r["unaided_hi_raw"]),
                     fmt_auc(r["aided_raw"], r["aided_lo_raw"], r["aided_hi_raw"]),
                     fmt_auc(r["es_raw"], r["es_lo_raw"], r["es_hi_raw"], cap=False),
                     fmt_p(r["pvalue_rrrc_raw"]), ""])
    cols = ["Variable", "Average AUC without CAD", "Average AUC with CAD", "Effect size",
            "P value, aided vs unaided", "P for interaction (unadjusted / adjusted)"]
    return pd.DataFrame(rows, columns=cols)


TABLE3_CAPTION = (
    "Table 3. Effect size for patient likelihood of malignancy (PLoM) by reader and "
    "index-nodule subgroup.\n"
    "Note - Average AUC (95% CI) and subgroup effect size (95% CI) are estimated by the"
    " DBM method; the column “P value, aided vs unaided” (subgroup rows) tests the "
    "aided vs unaided difference within each subgroup by the DBM method (unadjusted). "
    "The column “P for interaction” (factor header rows) tests whether the aided vs "
    "unaided effect size differs across that factor’s subgroups (the subgroup × "
    "treatment interaction), by one-way ANOVA on subgroup-level means of the "
    "per-(reader, case) DBM aided vs unaided pseudovalue differences. Adjusted P values"
    " use the Benjamini-Hochberg procedure across the 11 factors. The small group "
    "includes 12 index nodules <5 mm. All subgroup analyses are exploratory. AI = "
    "artificial intelligence, AUC = area under the receiver operating characteristic "
    "curve, CAD = computer-aided detection and diagnosis, DBM = Dorfman-Berbaum-Metz, "
    "LCP = Lung Cancer Prediction, PLoM = patient likelihood of malignancy."
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


# --------------------------------------------------------------------------- #
# Table E2 -- reader characteristics
# --------------------------------------------------------------------------- #
READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]

TABLE_E2_CAPTION = (
    "Table E2. Reader characteristics (n = 15). Years of independent CT interpretation "
    "reported.\n"
    "Note - LCS = Lung Cancer Screening. UK - United Kingdom; EU - Europe."
)


# Sub-specialty strings in participant_demographic.csv -> manuscript wording (Table E2).
E2_SPECIALTY_DISPLAY = {
    "General; Nuclear Medicine and Head & Neck Imaging": "General; Nuclear Medicine; Head & Neck",
    "Neuroradiology; Lung cancer screening; General": "Neuro; LCS; General",
    "Cardiothoracic; Head and Neck": "Cardiothoracic; Head & Neck",
}


def build_table_E2(csv_path: Path) -> pd.DataFrame:
    """Reader table in reader order; the institution is used for the region only and is
    not output. (The ``country`` column is not used: it disagrees with the institution
    for one reader.)"""
    d = pd.read_csv(csv_path, encoding="utf-8-sig").set_index("participant_id")
    missing = [r for r in READERS if r not in d.index]
    if missing:
        raise ValueError(f"readers missing from {csv_path.name}: {missing}")
    d = d.loc[READERS]
    years = d["exp_y"].map(lambda y: f"{y:g}")
    return pd.DataFrame({
        "Index": range(1, len(READERS) + 1),
        "Radiology sub-specialty(ies)": d["sub_specialty"].str.strip()
                                        .replace(E2_SPECIALTY_DISPLAY).values,
        "Years of Experience": years.values,
        "Country": d["practice"].str.strip().str.rsplit(",", n=1).str[-1].str.strip()
                  .map(lambda c: "UK" if c == "UK" else "EU").values,
        "Academic": d["academic"].str.strip().values,
    })


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = load_patient_gt(args.data_root)

    tables = [
        ("table1_demographics.csv", TABLE1_CAPTION, build_table1(patients)),
        ("table2_index_nodule.csv", TABLE2_CAPTION, build_table2(patients)),
        ("table3_effect_size_by_subgroup.csv", TABLE3_CAPTION,
         build_table3(args.data_root / "mrmc" / "dataset_effect_size_analysis.csv",
                      args.data_root / "mrmc" / "dataset_subgroup_interaction_analysis.csv")),
        ("tableE2_reader_characteristics.csv", TABLE_E2_CAPTION,
         build_table_E2(args.data_root / "dataset" / "participant_demographic.csv")),
    ]
    for fname, caption, df in tables:
        print(caption)
        print(df.to_string(index=False))
        out = args.out_dir / fname
        df.to_csv(out, index=False)
        print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
