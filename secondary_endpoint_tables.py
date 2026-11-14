"""Secondary endpoint tables for the CLEAREST main study manuscript (Radiology).

Reproduces the detection / agreement tables:

Table 4  -- Inter-reader agreement (Fleiss' kappa, 95% CI) with and without CAD,
            for five items: malignancy-risk category and clinical recommendation
            (patient-level), and nodule detection, size and attenuation
            (nodule-level, on matched detections). Each row also reports a paired
            patient-cluster bootstrap p-value for the aided-vs-unaided difference.
Table E1 -- Ground-truth counts for the main-study dataset by scan/nodule and
            cancer/benign split (supplemental material).
Table E5 -- Reader-averaged sensitivity and specificity at the pre-specified ACCP
            thresholds (5%, 65%), pLoM >= threshold, paired two-way bootstrap.
Table E4 -- Nodule-detection performance: across-reader mean sensitivity and
            false positives (FP) per case at a 1x match radius, unaided / aided /
            CAD alone, for all nodules and expert-annotated (EA) nodules >=5 mm
            (the tabular companion of Figure 5), plus reader-averaged sensitivity
            by nodule size (5 to <9, 9 to <15, >=15 mm long-axis diameter).

Each table is written as a CSV to the output directory (default: ``tables/``).

Data inputs (not distributed with this repository)
--------------------------------------------------
Under ``<data-root>`` (default: ``../resources/clearest_main_study``):
  final_exports/<reader>_S1_S2_final.csv                              (Table 4)
      Per-reader patient-level assessments (risk, recommendation) and per-nodule
      annotations (attenuation) used for the agreement rows.
  detection_rate/matched_all_readers_sessions_notn_all.csv           (Table 4)
      Matched detections per (reader, session) for the detection/size/attenuation
      agreement rows (1x-diameter match radius = diam_multiplier 0.75 rows used).
  mrmc/all/clearest_results_patient_lom.csv                           (Table E5)
      Per (reader, case) unaided (LOM1) and aided (LOM2) pLoM and truth L.
  dataset/clearest_patient_data_240_main_study_max_lcp_score_gt.csv   (Table E5)
      Case-level truth and nodule-negative flag (is_cancer, is_nodule_gt).
  final_exports/GT_extended_corrected.csv                            (Table E1)
      Nodule-level ground-truth export (one row per nodule / negative scan).
  detection_rate/matched_all_readers_sessions_notn_all_ignore_filtered.csv  (Table E4)
      Matched detections per (reader, session); ignore-list FPs removed.

Usage
-----
    python secondary_endpoint_tables.py \
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


READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]


# =========================================================================== #
# Table 4 -- inter-observer agreement (Fleiss' kappa)
# --------------------------------------------------------------------------- #
# The manuscript Table 4 combines five agreement items, with and without CAD:
#   Patient-level (agreement_paper.py):
#     - malignancy-risk category, derived from the categorized likelihood of
#       malignancy (pLoM) rather than the reader-assigned risk field
#     - clinical (management) recommendation (grouped follow-up)
#   Nodule-level, on matched detections (inter_reader_fleiss_kappa_table.py):
#     - nodule detection, size and attenuation
# =========================================================================== #
TABLE4_DM = 0.75  # match radius (diameter multiplier) for the nodule-level rows

# pLoM (%) risk categories: ACCP-based, as used by Kim et al, Radiology 2022;304:683-691
# (Table 4: very low <5%, low-moderate 5%-30%, moderate-high 31%-65%, high >65%; citing Gould et
# al, Chest 2013). pLoM is recorded in whole percent, so the half-open edges are 5, 31, 66.
# (Until 2026-11-11 Table 4 used <10, 10-35, 36-70, >=71%, edges [10, 36, 71].)
PLOM_BIN_EDGES = [5, 31, 66]
PLOM_BINS = [("Very low (<5%)", 0), ("Low-moderate (5%–30%)", 1),
             ("Moderate-high (31%–65%)", 2), ("High (>65%)", 3)]

# patient_follow_up_number 0..8 -> 3 management buckets:
#   No action (0,1) | CT surveillance (2,3,4) | Diagnostic procedure (5,6,7,8)
FOLLOWUP_GROUP_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
FU_GROUPS = [("No action", 0), ("CT surveillance", 1), ("Diagnostic procedure", 2)]

SIZE_CATEGORIES = ["small", "medium", "large"]
SIZE_DISPLAY = {"small": "Small (<9 mm)", "medium": "Medium (9 to <15 mm)", "large": "Large (≥15 mm)"}
ATTEN_CATEGORIES = ["solid", "part_solid", "ggo"]
ATTEN_DISPLAY = {"solid": "Solid", "part_solid": "Part-solid", "ggo": "Ground Glass"}


def fleiss_kappa(mat: np.ndarray, categories: list) -> float:
    """Fleiss' kappa for multi-rater categorical agreement.

    `categories` is the list of possible labels (integers or strings); ratings
    that are missing (NaN) or not in `categories` are excluded. Vectorized.
    """
    n_subjects, n_raters = mat.shape
    if n_subjects < 2 or n_raters < 2 or len(categories) < 2:
        return np.nan
    counts = np.stack([(mat == c).sum(axis=1) for c in categories], axis=1).astype(float)
    valid = counts.sum(axis=1)  # ratings that fall in a listed category
    keep = valid >= 2
    counts, valid = counts[keep], valid[keep]
    if counts.shape[0] < 2:
        return np.nan
    P_i = (np.sum(counts ** 2, axis=1) - valid) / (valid * (valid - 1))
    p_j = counts.sum(axis=0) / valid.sum()
    P_e = np.sum(p_j ** 2)
    return 1.0 if P_e == 1.0 else float((P_i.mean() - P_e) / (1 - P_e))


# -- patient-level (subject x reader) matrices -------------------------------
def _extract_patient_level(reader_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(reader_csv_path)
    cols = ["patient_lom", "patient_follow_up_number"]
    return df.groupby(["study_id", "is_aided_encord"])[cols].first().reset_index()


def build_reader_session_table(exports_dir: Path, reader_list, session: int) -> pd.DataFrame:
    frames = []
    for reader in reader_list:
        p = _extract_patient_level(f"{exports_dir}/{reader}_S1_S2_final.csv")
        sess = p[p["is_aided_encord"] == session].copy()
        sess["reader"] = reader
        frames.append(sess)
    return pd.concat(frames, ignore_index=True)


def _patient_matrix(session_df, field, reader_list):
    piv = session_df.pivot_table(index="study_id", columns="reader", values=field, aggfunc="first")
    for r in reader_list:
        if r not in piv.columns:
            piv[r] = np.nan
    return piv[reader_list].values


def _plom_to_bin(mat: np.ndarray) -> np.ndarray:
    """Bin a (subjects x raters) pLoM (0-100%) matrix into 4 ordered categories."""
    out = np.full_like(mat, np.nan, dtype=float)
    valid = ~np.isnan(mat)
    out[valid] = np.digitize(mat[valid], PLOM_BIN_EDGES, right=False).astype(float)
    return out


# -- nodule-level (GT-nodule x reader) matrices, from matched detections -----
def build_reader_annotations(exports_dir: Path, reader_list) -> pd.DataFrame:
    """Per-detection reader attenuation keyed by (key, reader, session, det_region_name)."""
    frames = []
    for reader in reader_list:
        df = pd.read_csv(f"{exports_dir}/{reader}_S1_S2_final.csv")
        df["key"] = df["study_id"]
        df["reader"] = reader
        for is_aided in (0, 1):
            sess = df[(df.is_aided_encord == is_aided) & (df.is_nodule == 1)].copy()
            sess["det_region_name"] = (
                "det" + sess.groupby("key").cumcount().add(1).astype(str).str.zfill(2))
            sess["reader_attenuation"] = sess["attenuation"].str.replace("part-solid", "part_solid")
            sess["reader_long_diameter"] = sess["long_diameter"]
            sess["reader_short_diameter"] = sess["short_diameter"]
            frames.append(sess[["key", "reader", "is_aided_encord", "det_region_name",
                                 "reader_attenuation", "reader_long_diameter",
                                 "reader_short_diameter"]])
    return pd.concat(frames, ignore_index=True)


def _detection_matrix(matched_df, reader_list):
    """Binary detection matrix over GT nodules: 1 = TP (detected), 0 = FN (missed)."""
    gt = matched_df[matched_df["type"].isin(["tp", "fn"])].copy()
    gt = gt[gt["reader"].isin(reader_list)]
    gt["detected"] = (gt["type"] == "tp").astype(int)
    mat = gt.pivot_table(index=["key", "region_name"], columns="reader",
                         values="detected", aggfunc="max")
    for r in reader_list:
        if r not in mat.columns:
            mat[r] = np.nan
    return mat[reader_list].values


def _category_matrix(tp_df, field, reader_list):
    piv = tp_df.pivot_table(index=["key", "region_name"], columns="reader",
                            values=field, aggfunc="first")
    for r in reader_list:
        if r not in piv.columns:
            piv[r] = np.nan
    return piv[reader_list].values


def _size_category(diameter):
    """Bin reader-measured long-axis diameter (mm, unrounded) into the prespecified
    groups: small <9, medium 9 to <15, large >=15 mm (half-open, as in Tables 2, 3, E1, E4)."""
    return pd.cut(diameter, bins=[-np.inf, 9, 15, np.inf],
                  labels=SIZE_CATEGORIES, right=False).astype("object")


def _binarize(mat, positive_label):
    """One-vs-rest: 1 where == label, 0 where another label, NaN where missing."""
    out = np.full(mat.shape, np.nan, dtype=float)
    notna = pd.notna(mat)
    out[notna] = (mat[notna] == positive_label).astype(float)
    return out


# -- bootstrap + formatting --------------------------------------------------
def _ci(boots):
    valid = boots[~np.isnan(boots)]
    if len(valid) < 10:
        return (np.nan, np.nan)
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def _diff_pvalue(boots_a, boots_u):
    diff = boots_a - boots_u
    valid = diff[~np.isnan(diff)]
    if len(valid) < 10:
        return np.nan
    return float(min(2 * min(np.mean(valid <= 0), np.mean(valid >= 0)), 1.0))


def _fmt_kappa(k, lo, hi):
    if not np.isfinite(k):
        return ""
    return f"{k:.2f} ({lo:.2f}, {hi:.2f})" if (np.isfinite(lo) and np.isfinite(hi)) else f"{k:.2f}"


def _fmt_p(p):
    """P value in Radiology style: no leading zero; two digits, three if < .01; '<.001' below .001."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<.001"
    return (f"{p:.3f}" if p < 0.01 else f"{p:.2f}").lstrip("0")


def _fmt_p_pair(p_raw, p_adj, at_floor):
    """'unadjusted / adjusted'. A row whose bootstrap P is 0 (no resample crossed zero) has a
    true P below the resolution of the bootstrap; its adjusted P is shown as an upper bound,
    '<' + the adjusted bound rounded up to three decimals."""
    if not np.isfinite(p_raw):
        return ""
    if at_floor:
        return f"<.001 / <{f'{np.ceil(p_adj * 1000 - 1e-9) / 1000:.3f}'.lstrip('0')}"
    return f"{_fmt_p(p_raw)} / {_fmt_p(p_adj)}"


def _header(label):
    return [label, "", "", np.nan]


TABLE4_CAPTION = (
    "Table 4. Inter-reader Agreement for Malignancy Risk category, Clinical "
    "Recommendation, nodule detection, and nodule size and attenuation, with and "
    "without CAD.\n"
    "Note - Cells are Fleiss' kappa (95% CI). Malignancy risk categories are derived "
    "from PLoM at the American College of Chest Physicians thresholds (24), as in Kim "
    "et al (7). Per-category rows use a one-vs-rest binary recoding so that each row "
    "quantifies agreement on a single category; the Overall row uses the full "
    "multi-category matrix (4 risk levels; 3 grouped recommendations). The 9 raw "
    "recommendation labels are grouped as follows - No action: no action (micro-nodules"
    " or no nodules) and no action (follow-up not required); CT surveillance: long-term"
    " (12 months), mid-term LDCT (6 months), and short-term LDCT (6 weeks-3 months) "
    "follow-up; Diagnostic procedure: nodule characterization PET/CT, biopsy "
    "techniques, surgical resection or other destructive techniques, and lung-cancer "
    "staging PET/CT. P value: two-sided bootstrap test on the paired difference (aided "
    "− unaided) in Fleiss’ kappa, resampling patients with replacement and keeping both"
    " reading conditions and all of a patient’s nodules together (paired patient-level "
    "bootstrap; 2000 resamples; readers fixed; percentile method); adjusted P values "
    "use the Benjamini-Hochberg procedure across the 18 comparisons, with an upper "
    "bound shown where P < .001; category-level rows are descriptive. CAD = "
    "computer-aided detection and diagnosis, LDCT = low-dose CT, PLoM = patient "
    "likelihood of malignancy."
)


# =========================================================================== #
# Table 4 bootstrap -- paired patient-cluster resampling
# --------------------------------------------------------------------------- #
# The 95% CIs and p-values of the kappa rows use a patient-level bootstrap:
#   1. Paired: each replicate draws ONE set of patients, and both the unaided and
#      the aided kappa are computed on it.
#   2. Patient clusters: nodule-level rows (detection, size, attenuation) keep all
#      of a drawn patient's nodules (nodules within a patient are not independent).
#      For size / attenuation each arm keeps its own true-positive nodules within
#      the drawn patients (paired at patient level).
#   3. Order-independent: the B patient draws are generated once, up front, and
#      reused by every row, so adding / reordering rows never changes other rows.
# Readers are held fixed (Fleiss' kappa is agreement within this reader panel).
# The resampling universe is all study patients, including those without nodules.
# =========================================================================== #
def _matrix_with_patients(piv, reader_list, patient_level):
    """(subjects x readers) values plus the patient id of each row of a pivot table."""
    for r in reader_list:
        if r not in piv.columns:
            piv[r] = np.nan
    patients = piv.index.get_level_values(patient_level).to_numpy()
    return piv[reader_list].values, patients


def _patient_matrix_ids(session_df, field, reader_list):
    piv = session_df.pivot_table(index="study_id", columns="reader", values=field, aggfunc="first")
    return _matrix_with_patients(piv, reader_list, "study_id")


def _detection_matrix_ids(matched_df, reader_list):
    gt = matched_df[matched_df["type"].isin(["tp", "fn"]) & matched_df["reader"].isin(reader_list)].copy()
    gt["detected"] = (gt["type"] == "tp").astype(int)
    piv = gt.pivot_table(index=["key", "region_name"], columns="reader", values="detected", aggfunc="max")
    return _matrix_with_patients(piv, reader_list, "key")


def _category_matrix_ids(tp_df, field, reader_list):
    piv = tp_df.pivot_table(index=["key", "region_name"], columns="reader", values=field, aggfunc="first")
    return _matrix_with_patients(piv, reader_list, "key")


class _PatientClusterBootstrap:
    """B patient-level resamples, drawn once and shared by every row of the table."""

    def __init__(self, patients, n_boot: int, seed: int):
        self.patients = np.asarray(sorted(set(patients)))
        self.pos = {p: i for i, p in enumerate(self.patients)}
        rng = np.random.default_rng(seed)
        n = len(self.patients)
        draws = rng.integers(0, n, size=(n_boot, n))
        # counts[b, j] = how many times patient j is drawn in replicate b
        self.counts = np.stack([np.bincount(d, minlength=n) for d in draws])

    def kappas(self, mat, row_patients, categories):
        """Bootstrap kappa per replicate: each row is repeated as often as its patient is drawn."""
        missing = set(row_patients) - set(self.pos)
        if missing:
            raise ValueError(f"{len(missing)} matrix rows belong to patients outside the universe")
        row_pos = np.array([self.pos[p] for p in row_patients], dtype=int)
        out = np.full(self.counts.shape[0], np.nan)
        for b, cnt in enumerate(self.counts):
            out[b] = fleiss_kappa(np.repeat(mat, cnt[row_pos], axis=0), categories)
        return out

    def row(self, label, u, a, categories):
        """Display row from (matrix, patients) pairs for the unaided (u) and aided (a) arm."""
        (mat_u, pat_u), (mat_a, pat_a) = u, a
        k_u, k_a = fleiss_kappa(mat_u, categories), fleiss_kappa(mat_a, categories)
        bu, ba = self.kappas(mat_u, pat_u, categories), self.kappas(mat_a, pat_a, categories)
        return [label, _fmt_kappa(k_u, *_ci(bu)), _fmt_kappa(k_a, *_ci(ba)), _diff_pvalue(ba, bu)]


def _binarize_ids(mp, positive_label):
    return _binarize(mp[0], positive_label), mp[1]


def build_table4(exports_dir: Path, matched_path: Path, reader_list,
                                n_boot: int, seed: int = 42) -> pd.DataFrame:
    """Combined inter-reader agreement table (patient-level + nodule-level), paired patient-cluster bootstrap."""
    sess = {s: build_reader_session_table(exports_dir, reader_list, s) for s in (0, 1)}
    for s in (0, 1):
        sess[s]["fu_group"] = sess[s]["patient_follow_up_number"].map(FOLLOWUP_GROUP_MAP)
    risk = {}
    for s in (0, 1):
        mat, pat = _patient_matrix_ids(sess[s], "patient_lom", reader_list)
        risk[s] = (_plom_to_bin(mat), pat)
    fu = {s: _patient_matrix_ids(sess[s], "fu_group", reader_list) for s in (0, 1)}

    matched = pd.read_csv(matched_path, low_memory=False)
    df = matched[(matched["reader"].isin(reader_list))
                 & (matched["diam_multiplier"] == TABLE4_DM)].copy()
    annotations = build_reader_annotations(exports_dir, reader_list)
    df = df.merge(annotations, on=["key", "reader", "is_aided_encord", "det_region_name"], how="left")

    universe = pd.concat([sess[0]["study_id"], sess[1]["study_id"], df["key"]]).unique()
    boot = _PatientClusterBootstrap(universe, n_boot, seed)
    print(f"  Table 4 paired patient-cluster bootstrap: {len(boot.patients)} patients, {n_boot} resamples")

    rows = [_header("Malignancy risk category")]
    for label, lvl in PLOM_BINS:
        rows.append(boot.row(f"  {label}", _binarize_ids(risk[0], lvl), _binarize_ids(risk[1], lvl), [0, 1]))
    rows.append(boot.row("  Overall", risk[0], risk[1], [0, 1, 2, 3]))

    rows.append(_header("Clinical recommendation"))
    for label, grp_val in FU_GROUPS:
        rows.append(boot.row(f"  {label}", _binarize_ids(fu[0], grp_val), _binarize_ids(fu[1], grp_val), [0, 1]))
    rows.append(boot.row("  Overall", fu[0], fu[1], [0, 1, 2]))

    rows.append(boot.row("Nodule detection", _detection_matrix_ids(df[df.is_aided_encord == 0], reader_list),
                         _detection_matrix_ids(df[df.is_aided_encord == 1], reader_list), [0, 1]))

    tp = df[df["type"] == "tp"].copy()
    join_ok = float((np.abs(tp["det_diameter"] - tp["reader_long_diameter"]) < 0.05).mean())
    print(f"  Table 4 join check: {join_ok:.4f} of TP rows have |det_diameter - reader_long_diameter| < 0.05")
    if join_ok < 0.99:
        print("  WARNING: Table 4 join check below 0.99; reader diameters may be misaligned")
    tp["size_category"] = _size_category(tp["det_diameter"])
    tp_u, tp_a = tp[tp.is_aided_encord == 0], tp[tp.is_aided_encord == 1]

    rows.append(_header("Nodule Size"))
    size_u = _category_matrix_ids(tp_u, "size_category", reader_list)
    size_a = _category_matrix_ids(tp_a, "size_category", reader_list)
    for cat in SIZE_CATEGORIES:
        rows.append(boot.row(f"  {SIZE_DISPLAY[cat]}", _binarize_ids(size_u, cat), _binarize_ids(size_a, cat), [0, 1]))
    rows.append(boot.row("  Overall", size_u, size_a, SIZE_CATEGORIES))

    rows.append(_header("Attenuation"))
    att_u = _category_matrix_ids(tp_u, "reader_attenuation", reader_list)
    att_a = _category_matrix_ids(tp_a, "reader_attenuation", reader_list)
    for cat in ATTEN_CATEGORIES:
        rows.append(boot.row(f"  {ATTEN_DISPLAY[cat]}", _binarize_ids(att_u, cat), _binarize_ids(att_a, cat), [0, 1]))
    rows.append(boot.row("  Overall", att_u, att_a, ATTEN_CATEGORIES))

    # Benjamini-Hochberg across the tested rows (18 comparisons). A bootstrap P of 0 is
    # replaced by the smallest resolvable two-sided P, 2 / B, before adjustment, and that
    # row's adjusted P is reported as an upper bound.
    floor = 2.0 / n_boot
    tested = [i for i, r in enumerate(rows) if np.isfinite(r[3])]
    assert len(tested) == 18, len(tested)
    raw = np.array([rows[i][3] for i in tested])
    adj = false_discovery_control(np.maximum(raw, floor), method="bh")
    for i, p_raw, p_adj in zip(tested, raw, adj):
        rows[i][3] = _fmt_p_pair(p_raw, p_adj, at_floor=p_raw < floor)
    for r in rows:
        if not isinstance(r[3], str):
            r[3] = ""
    return pd.DataFrame(rows, columns=["Variable", "Without CAD", "With CAD",
                                       "P (unadjusted / adjusted)"])


# =========================================================================== #
# Table E5 -- sensitivity and specificity at malignancy-risk thresholds
# --------------------------------------------------------------------------- #
# Patient-level pLoM (0-100%) read without (LOM1) and with (LOM2) CAD, dichotomized at the
# ACCP thresholds (Gould et al, Chest 2013; as in Kim et al, Radiology 2022), 5% and 65% --
# the pre-specified secondary endpoint.
# A read is positive if pLoM >= threshold (the pre-specified rule; it reproduces the
# original secondary-endpoint analysis). Readers often score exactly 65%, so these reads
# count as positive.
# Sensitivity uses the 95 malignant cases. Specificity is shown for all 145 non-malignant
# cases (93 benign nodules + 52 nodule-negative CTs) and for the 93 benign nodules only.
# Values are reader-averaged (mean of the 15 per-reader proportions).
#
# Inference: paired two-way (random-reader, random-case) percentile bootstrap, matching the
# RRRC generalization of the primary endpoint. Each replicate draws readers with
# replacement and patients with replacement within diagnosis stratum (malignant, benign
# nodule, nodule-negative), and evaluates BOTH arms on the same readers and patients.
# P = two-sided bootstrap test of the aided - unaided difference.
# =========================================================================== #
E5_THRESHOLDS = [("ACCP (pre-specified)", 5), ("ACCP (pre-specified)", 65)]


def _fmt_pct(est, lo, hi):
    return f"{est:.1f} ({lo:.1f}, {hi:.1f})"


def _two_way_draws(n_readers, n_cases, strata_masks, n_boot, seed):
    """Paired two-way bootstrap draws shared by both reading conditions.

    Returns reader counts (B x R; readers drawn with replacement) and case counts
    (B x C; cases drawn with replacement within each stratum mask, in order)."""
    rng = np.random.default_rng(seed)
    r_cnt = np.stack([np.bincount(rng.integers(0, n_readers, n_readers), minlength=n_readers)
                      for _ in range(n_boot)])
    c_cnt = np.zeros((n_boot, n_cases))
    for mask in strata_masks:
        idx = np.flatnonzero(mask)
        for b in range(n_boot):
            c_cnt[b] += np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=n_cases)
    return r_cnt, c_cnt


def _reader_avg_boot(indicator, subset, r_cnt, c_cnt):
    """Reader-averaged % of an (R x C) 0/1 indicator over `subset` cases: point estimate
    (mean of per-reader proportions) and its bootstrap replicates."""
    sub = indicator[:, subset].astype(float)
    point = 100 * sub.mean(axis=1).mean()
    w = c_cnt[:, subset]                                      # B x C_sub
    per_reader = (w @ sub.T) / w.sum(axis=1, keepdims=True)   # B x R
    boot = 100 * (per_reader * r_cnt).sum(axis=1) / r_cnt.sum(axis=1)
    return point, boot


def _boot_p(diff):
    """Two-sided bootstrap P that the quantity is zero."""
    return min(2 * min(np.mean(diff <= 0), np.mean(diff >= 0)), 1.0)


def build_table_E5(lom_csv: Path, gt_csv: Path, n_boot: int, seed: int = 42) -> pd.DataFrame:
    lom = pd.read_csv(lom_csv)
    gt = pd.read_csv(gt_csv, usecols=["study_id_stage1", "is_cancer", "is_nodule_gt"])
    lom = lom.merge(gt, left_on="Case", right_on="study_id_stage1", how="left", validate="many_to_one")
    if lom["is_cancer"].isna().any():
        raise ValueError("cases in the LOM file without ground truth")
    if not (lom["is_cancer"] == lom["L"]).all():
        raise ValueError("truth label L disagrees with is_cancer")

    readers = np.sort(lom["Reader"].unique())
    cases = lom.groupby("Case")[["L", "is_nodule_gt"]].first()
    unaided = lom.pivot(index="Reader", columns="Case", values="LOM1").loc[readers, cases.index].values
    aided = lom.pivot(index="Reader", columns="Case", values="LOM2").loc[readers, cases.index].values
    if np.isnan(unaided).any() or np.isnan(aided).any():
        raise ValueError("missing reads: every reader must read every case in both arms")
    mal = cases["L"].values == 1
    ben_nod = (~mal) & (cases["is_nodule_gt"].values == 1)
    strata = {"malignant": mal, "benign nodule": ben_nod, "nodule-negative": (~mal) & ~ben_nod}
    n_r, n_c = unaided.shape
    print(f"  Table E5: {n_r} readers, {n_c} cases "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in strata.items()))

    r_cnt, c_cnt = _two_way_draws(n_r, n_c, strata.values(), n_boot, seed)

    def reader_avg(correct, subset):
        return _reader_avg_boot(correct, subset, r_cnt, c_cnt)

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    rows = []
    for guideline, th in E5_THRESHOLDS:
        pos_u, pos_a = unaided >= th, aided >= th
        metrics = [("sensitivity", f"malignant (n = {mal.sum()})", mal, True),
                   ("specificity", f"all non-malignant (n = {(~mal).sum()})", ~mal, False),
                   ("specificity", f"benign nodules (n = {ben_nod.sum()})", ben_nod, False)]
        for metric, population, subset, is_sens in metrics:
            cu, ca = (pos_u, pos_a) if is_sens else (~pos_u, ~pos_a)
            pu, bu = reader_avg(cu, subset)
            pa, ba = reader_avg(ca, subset)
            d = ba - bu
            p = _boot_p(d)
            print(f"    {guideline} >={th}% {metric}, {population}: difference "
                  f"{_fmt_pct(pa - pu, *ci(d))}")
            rows.append({
                "Measure": f"≥{th}% : {metric}, {population}",
                "Without CAD": _fmt_pct(pu, *ci(bu)),
                "With CAD": _fmt_pct(pa, *ci(ba)),
                "P value": _fmt_p(p),
            })
    return pd.DataFrame(rows)


TABLE_E5_CAPTION = (
    "Table E5. Patient-level sensitivity and specificity at the American College of "
    "Chest Physicians (ACCP) 5% and 65% thresholds (pre-specified), without and with "
    "CAD.\n"
    "Note - Reader-averaged percentages (95% CI). A read was positive if the patient "
    "likelihood of malignancy was at or above the threshold (pre-specified rule). "
    "Sensitivity is computed in the 95 malignant examinations; specificity in all 145 "
    "non-malignant examinations and, separately, in the 93 examinations with a benign "
    "nodule. Differences are with minus without CAD, in percentage points. CIs and P "
    "values by a paired two-way percentile bootstrap (2,000 resamples; readers "
    "resampled and examinations resampled within diagnosis stratum, with both reading "
    "conditions on the same resample)."
)


# =========================================================================== #
# Change in clinical recommendation with CAD, CTs with a nodule only
# --------------------------------------------------------------------------- #
# Per read, the 9-level follow-up recommendation (patient_follow_up_number 0-8, ordered
# by intensity of management) without and with CAD. A change is "correct" if management
# becomes more intensive for a malignant case or less intensive for a benign case, and
# "incorrect" in the opposite direction. Per reader: % of cases with a correct change,
# % with an incorrect change, and the net (correct - incorrect); values are reader-averaged.
# Restricted to CTs with a nodule: 95 malignant and 93 benign-nodule cases (the 52
# nodule-negative CTs are excluded). Same paired two-way bootstrap as Table E5 (readers
# and cases resampled; cases within malignant / benign-nodule strata); P tests net = 0.
# (The original analysis, secondary_outcomes/output/change_in_recommendation*.csv, used all
# 145 non-malignant cases and a case-only bootstrap.)
# =========================================================================== #
def build_table_recommendation_change(exports_dir: Path, gt_csv: Path, reader_list,
                                      n_boot: int, seed: int = 42) -> pd.DataFrame:
    gt = pd.read_csv(gt_csv, usecols=["study_id_stage1", "is_cancer", "is_nodule_gt"])
    gt = gt[gt["is_nodule_gt"] == 1].set_index("study_id_stage1")
    unaided, aided = [], []
    for reader in reader_list:
        d = pd.read_csv(f"{exports_dir}/{reader}_S1_S2_final.csv")
        rec = d.groupby(["study_id", "is_aided_encord"])["patient_follow_up_number"].first().unstack()
        rec = rec.reindex(gt.index)
        unaided.append(rec[0].values)
        aided.append(rec[1].values)
    unaided, aided = np.array(unaided, dtype=float), np.array(aided, dtype=float)
    if np.isnan(unaided).any() or np.isnan(aided).any():
        raise ValueError("missing recommendations for CTs with a nodule")
    mal = gt["is_cancer"].values == 1
    ben = ~mal
    n_r, n_c = unaided.shape
    print(f"  Recommendation change: {n_r} readers, {mal.sum()} malignant + {ben.sum()} benign-nodule CTs")
    r_cnt, c_cnt = _two_way_draws(n_r, n_c, [mal, ben], n_boot, seed)

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    rows = []
    for label, subset, sign in [(f"Malignant (n={mal.sum()})", mal, 1),
                                (f"Benign nodule (n={ben.sum()})", ben, -1)]:
        change = sign * (aided - unaided)       # > 0 = change in the correct direction
        pc, bc = _reader_avg_boot(change > 0, subset, r_cnt, c_cnt)
        pi, bi = _reader_avg_boot(change < 0, subset, r_cnt, c_cnt)
        rows.append({
            "Cases": label,
            "Correct change": _fmt_pct(pc, *ci(bc)),
            "Incorrect change": _fmt_pct(pi, *ci(bi)),
            "Net change": _fmt_pct(pc - pi, *ci(bc - bi)),
            "P (net = 0)": _fmt_p(_boot_p(bc - bi)),
        })
    return pd.DataFrame(rows)


RECOMMENDATION_CHANGE_CAPTION = (
    "Change in clinical recommendation with CAD, CTs with a nodule only (95 malignant, 93 "
    "benign nodules; 52 nodule-negative CTs excluded). Reader-averaged % of cases (95% CI) whose "
    "9-level follow-up recommendation changed in the correct direction (more intensive for "
    "malignant, less intensive for benign), in the incorrect direction, and the net change "
    "(correct - incorrect). 95% CIs and two-sided P (net = 0) from a paired two-way percentile "
    "bootstrap (2,000 resamples; readers and cases resampled, cases within diagnosis stratum)."
)


# =========================================================================== #
# Table E1 -- ground-truth counts
# =========================================================================== #
def _log_mhg_prob(table):
    t = np.asarray(table, dtype=np.int64)
    rs, cs, n = t.sum(axis=1), t.sum(axis=0), int(t.sum())
    return (sum(lgamma(int(r) + 1) for r in rs) + sum(lgamma(int(c) + 1) for c in cs)
            - lgamma(n + 1) - sum(lgamma(int(x) + 1) for x in t.flatten()))


def fisher_freeman_halton_3x2(table):
    """Exact Fisher-Freeman-Halton two-sided p-value for a 3x2 table."""
    t = np.asarray(table, dtype=np.int64)
    assert t.shape == (3, 2), f"expected 3x2 table, got {t.shape}"
    rs, cs = t.sum(axis=1), t.sum(axis=0)
    log_p_obs = _log_mhg_prob(t)
    log_ps = []
    for a in range(min(int(rs[0]), int(cs[0])) + 1):
        for b in range(max(0, int(cs[0]) - a - int(rs[2])), min(int(rs[1]), int(cs[0]) - a) + 1):
            c = int(cs[0]) - a - b
            log_ps.append(_log_mhg_prob(np.array(
                [[a, rs[0] - a], [b, rs[1] - b], [c, rs[2] - c]], dtype=np.int64)))
    log_ps = np.array(log_ps)
    mask = log_ps <= log_p_obs + 1e-12
    m = log_ps[mask].max()
    return float(np.exp(m) * np.exp(log_ps[mask] - m).sum())


def build_table_E1(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(csv_path)

    def map_size_local(s):
        if pd.isna(s):
            return np.nan
        return "small" if s < 9 else "medium" if s < 15 else "large"

    # Prespecified size groups on the unrounded long-axis diameter.
    gt["size_bucket"] = gt.diameter.apply(map_size_local)
    gt["easyhard"] = gt.case_difficulty.map({
        "0_-_benign_easy": "easy", "2_-_malignant_easy": "easy",
        "1_-_benign_hard": "hard", "3_-_malignant_hard": "hard"})

    nod = gt[gt.is_nodule == 1]
    nod_can, nod_ben = nod[nod.is_cancer == 1], nod[nod.is_cancer == 0]

    g = gt.groupby("study_id")
    scan = pd.DataFrame({"n_nod": g.is_nodule.sum(), "any_cancer": g.is_cancer.max(),
                         "eh": g.easyhard.first()})
    scan["has_nod"] = scan.n_nod > 0
    scan_neg, scan_pos = scan[~scan.has_nod], scan[scan.has_nod]
    scan_can, scan_ben = scan_pos[scan_pos.any_cancer == 1], scan_pos[scan_pos.any_cancer == 0]

    SD = {"tot": len(scan), "neg": len(scan_neg), "nd": len(scan_pos),
          "can": len(scan_can), "ben": len(scan_ben)}
    ND = {"tot": len(nod), "neg": 0, "nd": len(nod), "can": len(nod_can), "ben": len(nod_ben)}
    order = ["tot", "neg", "nd", "can", "ben"]

    def cell(n, d):
        return "0 (-)" if d == 0 else f"{int(n)} ({round(100 * n / d)})"

    def fmt_p(p):
        return "" if p == "" or pd.isna(p) else _fmt_p(p)

    def row(label, counts, denom, p=""):
        return [label] + [cell(counts[c], denom[c]) for c in order] + [fmt_p(p)]

    def nod_counts(series, key):
        return {"tot": (series == key).sum(), "neg": 0, "nd": (series == key).sum(),
                "can": (nod_can[series.name] == key).sum(),
                "ben": (nod_ben[series.name] == key).sum()}

    def chi2_p(can_grp, ben_grp, keys):
        tab = np.array([[(can_grp == k).sum(), (ben_grp == k).sum()] for k in keys])
        return stats.chi2_contingency(tab)[1]

    def section(label, p=""):
        return [label, "", "", "", "", "", fmt_p(p)]

    # Cancer-vs-Benign tests (chi-square; Fisher-Freeman-Halton exact for attenuation).
    p_size = chi2_p(nod_can.size_bucket, nod_ben.size_bucket, ["small", "medium", "large"])
    att_tab = np.array([[(nod_can.attenuation == k).sum(), (nod_ben.attenuation == k).sum()]
                        for k in ["solid", "part-solid", "ggo"]])
    p_att = fisher_freeman_halton_3x2(att_tab)
    p_marg = chi2_p(nod_can.is_spiculated, nod_ben.is_spiculated, ["no", "yes"])
    p_diff = chi2_p(scan_can.eh, scan_ben.eh, ["easy", "hard"])

    rows = []
    rows.append(row("CT scans",
                    {"tot": SD["tot"], "neg": SD["neg"], "nd": SD["nd"], "can": SD["can"], "ben": SD["ben"]},
                    {c: SD["tot"] for c in order}))
    # Difficulty is per CT scan; the other characteristics are per nodule.
    rows.append(section("Difficulty", p_diff))
    for label, key in [("  Easy", "easy"), ("  Hard", "hard")]:
        counts = {"tot": (scan.eh == key).sum(), "neg": (scan_neg.eh == key).sum(),
                  "nd": (scan_pos.eh == key).sum(), "can": (scan_can.eh == key).sum(),
                  "ben": (scan_ben.eh == key).sum()}
        rows.append(row(label, counts, SD))

    rows.append(row("Nodules",
                    {"tot": ND["tot"], "neg": 0, "nd": ND["nd"], "can": ND["can"], "ben": ND["ben"]},
                    {c: ND["tot"] for c in order}))

    rows.append(section("long-axis diameter", p_size))
    for label, key in [("  Small (<9 mm)", "small"), ("  Medium (9 to <15 mm)", "medium"),
                       ("  Large (≥15 mm)", "large")]:
        rows.append(row(label, nod_counts(nod.size_bucket, key), ND))

    rows.append(section("Attenuation", p_att))
    for label, key in [("  Solid", "solid"), ("  Part-solid", "part-solid"), ("  GGO", "ggo")]:
        rows.append(row(label, nod_counts(nod.attenuation, key), ND))

    rows.append(section("Margin", p_marg))
    for label, key in [("  Non-spiculated", "no"), ("  Spiculated", "yes")]:
        rows.append(row(label, nod_counts(nod.is_spiculated, key), ND))

    cols = ["Variable", f"Total (n={SD['tot']})", f"NN (n={SD['neg']})", f"IPN (n={SD['nd']})",
            f"Cancer (n={SD['can']})", f"Benign (n={SD['ben']})", "P value"]
    return pd.DataFrame(rows, columns=cols)


TABLE_E1_CAPTION = (
    "Table E1. Expert adjudication results for nodule detection and characterization "
    "(size as long-axis diameter, attenuation, margin) and level of difficulty.\n"
    "Note - Expert adjudication nodule/characteristics counts for the main-study "
    "dataset. Cell values are n (column %). 'Negative - nodule': CT scans with no "
    "annotated nodule; IPN - scans/nodules with Indeterminate Pulmonary Nodule "
    "annotations (Cancer and Benign). P value (Cancer vs Benign), shown on each "
    "characteristic's header row: Pearson chi-square test of independence; attenuation, "
    "Fisher-Freeman-Halton exact test. Size groups "
    "use long-axis diameter; the small group includes nodules <5 mm."
)


# =========================================================================== #
# Table E4 -- nodule-detection performance (tabular companion of Figure 5)
# =========================================================================== #
OPERATING_DM = 1
MIN_DIAMETER_MM = 5


def _detection_perf(matched_op, cases, gt_min_diam=None):
    tp_fn = matched_op[matched_op["type"].isin(["tp", "fn"])]
    if gt_min_diam is not None:
        tp_fn = tp_fn[tp_fn["gt_diameter"] >= gt_min_diam]
    fp = matched_op[matched_op["type"] == "fp"]
    counts = (tp_fn.groupby(["reader", "is_aided_encord"])["type"]
              .value_counts().unstack(fill_value=0).reset_index())
    for col in ("tp", "fn"):
        if col not in counts.columns:
            counts[col] = 0
    counts["sensitivity"] = counts["tp"] / (counts["tp"] + counts["fn"])
    fp_counts = fp.groupby(["reader", "is_aided_encord"]).size().rename("fp").reset_index()
    out = counts.merge(fp_counts, on=["reader", "is_aided_encord"], how="left").fillna({"fp": 0})
    out = out.merge(cases, on=["reader", "is_aided_encord"], how="left")
    out["fp_per_case"] = out["fp"] / out["n_cases"]
    return out


def _mean_se(sub, col):
    return sub[col].mean(), sub[col].std(ddof=1) / np.sqrt(len(sub))


def build_table_E4(matched: pd.DataFrame) -> pd.DataFrame:
    """Across-reader mean sensitivity and FP/case (unaided/aided/CAD alone)."""
    cases = (matched.groupby(["reader", "is_aided_encord"])["key"]
             .nunique().rename("n_cases").reset_index())
    matched_op = matched[matched["diam_multiplier"] == OPERATING_DM]

    def fmt(mean, se):
        return f"{mean:.3f} ({se:.3f})"

    rows = []
    for subset_label, gt_min in [("All nodules", None), (f"EA >= {MIN_DIAMETER_MM} mm", MIN_DIAMETER_MM)]:
        perf = _detection_perf(matched_op, cases, gt_min_diam=gt_min)
        readers = perf[perf["reader"].isin(READERS)]
        ai = perf[perf["reader"] == "ai"]
        for session, arm in [("Unaided", readers[readers.is_aided_encord == 0]),
                             ("Aided", readers[readers.is_aided_encord == 1])]:
            sens_m, sens_se = _mean_se(arm, "sensitivity")
            fpc_m, fpc_se = _mean_se(arm, "fp_per_case")
            rows.append([subset_label, session, len(arm), fmt(sens_m, sens_se), fmt(fpc_m, fpc_se)])
        if len(ai):
            rows.append([subset_label, "CAD alone", 1,
                         f"{ai['sensitivity'].iloc[0]:.3f}", f"{ai['fp_per_case'].iloc[0]:.3f}"])

    cols = ["Subset", "Session", "n_readers", "Sensitivity (SE)", "FP per case (SE)"]
    return pd.DataFrame(rows, columns=cols)


TABLE_E4_CAPTION = (
    "Nodule-detection performance at a 1x nodule-diameter match radius (tabular "
    "companion of Figure 5; the sensitivity rows also appear in Table E4): "
    "across-reader mean sensitivity and false positives (FP) per case (standard error),"
    " without CAD, with CAD and CAD alone, for all nodules and expert-adjudicated (EA) "
    "nodules >=5 mm. The >=5 mm size filter applies to TP/FN only (standard FROC "
    "convention)."
)


def _sens_counts(tp_fn: pd.DataFrame) -> pd.DataFrame:
    """Per-(reader, arm) TP/FN counts, sensitivity and number of GT nodules."""
    counts = (tp_fn.groupby(["reader", "is_aided_encord"])["type"]
              .value_counts().unstack(fill_value=0).reset_index())
    for col in ["tp", "fn"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["sensitivity"] = counts["tp"] / (counts["tp"] + counts["fn"])
    counts["n_gt"] = counts["tp"] + counts["fn"]
    return counts


def build_table_E4_by_size(matched: pd.DataFrame) -> pd.DataFrame:
    """Reader-averaged detection sensitivity by nodule-size group (submitted Table E4 layout).

    The size groups use the GT long axis (``gt_diameter``, half-open, unrounded;
    small 5 to <9 mm).
    """
    matched_op = matched[matched["diam_multiplier"] == OPERATING_DM]
    tp_fn = matched_op[matched_op["type"].isin(["tp", "fn"])].copy()

    groups = [
        ("All (includes <5 mm)", lambda t: np.ones(len(t), dtype=bool)),
        ("Nodules ≥ 5 mm", lambda t: (t["gt_diameter"] >= 5).values),
        # Same 9 and 15 mm cut-offs as Tables 2, 3, 4, E1; detection starts at 5 mm (the
        # ">= 5 mm" row and Figure 5), so small is 5 to <9 mm here.
        ("Small (5 to <9 mm)", lambda t: ((t["gt_diameter"] >= 5) & (t["gt_diameter"] < 9)).values),
        ("Medium (9 to <15 mm)", lambda t: ((t["gt_diameter"] >= 9) & (t["gt_diameter"] < 15)).values),
        ("Large (≥ 15 mm)", lambda t: (t["gt_diameter"] >= 15).values),
    ]

    cols = ["Group", "CAD-alone", "# nodules", "Without CAD", "With CAD", "P value"]
    rows = []
    for label, mask_fn in groups:
        perf = _sens_counts(tp_fn[mask_fn(tp_fn)])
        rdr = perf[perf["reader"].isin(READERS)]
        ai = perf[perf["reader"] == "ai"]
        unaided = rdr[rdr["is_aided_encord"] == 0]
        aided = rdr[rdr["is_aided_encord"] == 1]

        def mean_se(sub, col):
            return sub[col].mean(), sub[col].std(ddof=1) / np.sqrt(len(sub))

        n_m, n_se = mean_se(unaided, "n_gt")
        cad = ai["tp"].sum() / (ai["tp"].sum() + ai["fn"].sum()) if len(ai) else np.nan
        wo_m, wo_se = mean_se(unaided, "sensitivity")
        w_m, w_se = mean_se(aided, "sensitivity")
        wide = rdr.pivot(index="reader", columns="is_aided_encord",
                         values="sensitivity").dropna()
        try:
            pval = stats.wilcoxon(wide[1].values, wide[0].values)[1]
        except ValueError:
            pval = np.nan  # all paired differences are zero
        rows.append({
            "Group": label,
            "CAD-alone": f"{cad:.2f}",
            "# nodules": f"{n_m:.1f} ({n_se:.1f})",
            "Without CAD": f"{wo_m:.2f} ({wo_se:.2f})",
            "With CAD": f"{w_m:.2f} ({w_se:.2f})",
            "P value": _fmt_p(pval) if np.isfinite(pval) else "",
        })
    return pd.DataFrame(rows, columns=cols)


TABLE_E4_BY_SIZE_CAPTION = (
    "Table E4. Reader-averaged nodule detection sensitivity stratified by size (1× "
    "matching radius).\n"
    "Note - Data reported as reader-average (Standard Error), except for CAD-alone. "
    "Average nodule detection sensitivity at a 1× nodule-diameter matching radius "
    "across the 15 readers (mean and Standard Error, over reader and session as "
    "indicated). The 31 sub-5 mm nodules (~10% of the cohort) cap overall AI-alone "
    "sensitivity at 0.90; on the reportable ≥ 5 mm subset CAD-alone sensitivity is "
    "0.85."
)


# =========================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CLEAREST_DATA_ROOT", "../resources/clearest_main_study")),
        help="Path to the clearest_main_study data directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("tables"),
                        help="Directory to write the table CSVs into.")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples for the Table 4 kappa CIs / p-values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = args.data_root / "final_exports"
    detection_dir = args.data_root / "detection_rate"

    print(TABLE4_CAPTION)
    table4 = build_table4(
        exports_dir, detection_dir / "matched_all_readers_sessions_notn_all.csv",
        READERS, n_boot=args.n_boot)
    print(table4.to_string(index=False))
    table4.to_csv(args.out_dir / "table4_agreement.csv", index=False)
    print(f"  wrote {args.out_dir / 'table4_agreement.csv'}\n")

    print(TABLE_E5_CAPTION)
    table_e5 = build_table_E5(
        args.data_root / "mrmc" / "all" / "clearest_results_patient_lom.csv",
        args.data_root / "dataset" / "clearest_patient_data_240_main_study_max_lcp_score_gt.csv",
        n_boot=args.n_boot)
    print(table_e5.to_string(index=False))
    table_e5.to_csv(args.out_dir / "tableE5_threshold_sensitivity_specificity.csv", index=False)
    print(f"  wrote {args.out_dir / 'tableE5_threshold_sensitivity_specificity.csv'}\n")

    print(RECOMMENDATION_CHANGE_CAPTION)
    rec_change = build_table_recommendation_change(
        exports_dir, args.data_root / "dataset" / "clearest_patient_data_240_main_study_max_lcp_score_gt.csv",
        READERS, n_boot=args.n_boot)
    print(rec_change.to_string(index=False))
    rec_change.to_csv(args.out_dir / "recommendation_change_nodule_cts.csv", index=False)
    print(f"  wrote {args.out_dir / 'recommendation_change_nodule_cts.csv'}\n")

    print(TABLE_E1_CAPTION)
    table_e1 = build_table_E1(exports_dir / "GT_extended_corrected.csv")
    print(table_e1.to_string(index=False))
    table_e1.to_csv(args.out_dir / "tableE1_ground_truth_counts.csv", index=False)
    print(f"  wrote {args.out_dir / 'tableE1_ground_truth_counts.csv'}\n")

    print(TABLE_E4_CAPTION)
    matched = pd.read_csv(
        detection_dir / "matched_all_readers_sessions_notn_all_ignore_filtered.csv",
        low_memory=False)
    table_e4 = build_table_E4(matched)
    print(table_e4.to_string(index=False))
    table_e4.to_csv(args.out_dir / "tableE4_detection_performance.csv", index=False)
    print(f"  wrote {args.out_dir / 'tableE4_detection_performance.csv'}\n")

    print(TABLE_E4_BY_SIZE_CAPTION)
    table_e4_size = build_table_E4_by_size(matched)
    print(table_e4_size.to_string(index=False))
    table_e4_size.to_csv(args.out_dir / "tableE4_sensitivity_by_size.csv", index=False)
    print(f"  wrote {args.out_dir / 'tableE4_sensitivity_by_size.csv'}\n")


if __name__ == "__main__":
    main()
