"""Secondary endpoint tables for the CLEAREST main study manuscript (Radiology).

Reproduces the detection / agreement tables:

Table 4  -- Inter-reader agreement (Fleiss' kappa, 95% CI) with and without CAD,
            for five items: malignancy-risk category and clinical recommendation
            (patient-level), and nodule detection, size and attenuation
            (nodule-level, on matched detections). Each row also reports a paired
            bootstrap p-value for the aided-vs-unaided difference.
Table E1 -- Ground-truth counts for the main-study dataset by scan/nodule and
            cancer/benign split (supplemental material).
Table E4 -- Nodule-detection performance: across-reader mean sensitivity and
            false positives (FP) per case at a 1x match radius, unaided / aided /
            CAD alone, for all nodules and expert-annotated (EA) nodules >=5 mm
            (the tabular companion of Figure 6; supplemental material).

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

# pLoM (%) risk bins, half-open (lower inclusive): <10, 10-35, 36-70, >=71.
PLOM_BIN_EDGES = [10, 36, 71]
PLOM_BINS = [("Very low (<10%)", 0), ("Low-moderate (10-35%)", 1),
             ("Moderate-high (36-70%)", 2), ("High (>=71%)", 3)]

# patient_follow_up_number 0..8 -> 3 management buckets:
#   No action (0,1) | CT surveillance (2,3,4) | Diagnostic procedure (5,6,7,8)
FOLLOWUP_GROUP_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
FU_GROUPS = [("No action", 0), ("CT surveillance", 1), ("Diagnostic procedure", 2)]

SIZE_CATEGORIES = ["small", "medium", "large"]
SIZE_DISPLAY = {"small": "Small", "medium": "Medium", "large": "Large"}
ATTEN_CATEGORIES = ["solid", "part_solid", "ggo"]
ATTEN_DISPLAY = {"solid": "Solid", "part_solid": "Part-solid", "ggo": "GGO"}


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
            frames.append(sess[["key", "reader", "is_aided_encord",
                                 "det_region_name", "reader_attenuation"]])
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
    """Bin reader-measured long-axis diameter (mm): small <=8, medium (8,15], large >15."""
    return pd.cut(diameter, bins=[-np.inf, 8, 15, np.inf],
                  labels=SIZE_CATEGORIES, right=True).astype("object")


def _binarize(mat, positive_label):
    """One-vs-rest: 1 where == label, 0 where another label, NaN where missing."""
    out = np.full(mat.shape, np.nan, dtype=float)
    notna = pd.notna(mat)
    out[notna] = (mat[notna] == positive_label).astype(float)
    return out


# -- bootstrap + formatting --------------------------------------------------
def _boot_kappa(mat, categories, n_boot, rng):
    n = mat.shape[0]
    boots = np.full(n_boot, np.nan)
    if n < 2:
        return boots
    for b in range(n_boot):
        boots[b] = fleiss_kappa(mat[rng.integers(0, n, size=n)], categories)
    return boots


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
    return "" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _kappa_row(label, mat_u, mat_a, categories, n_boot, rng):
    """One display row: label, W/o CAD kappa (CI), With CAD kappa (CI), p-value."""
    k_u, k_a = fleiss_kappa(mat_u, categories), fleiss_kappa(mat_a, categories)
    bu, ba = _boot_kappa(mat_u, categories, n_boot, rng), _boot_kappa(mat_a, categories, n_boot, rng)
    return [label, _fmt_kappa(k_u, *_ci(bu)), _fmt_kappa(k_a, *_ci(ba)), _fmt_p(_diff_pvalue(ba, bu))]


def _header(label):
    return [label, "", "", ""]


def build_table4(exports_dir: Path, matched_path: Path, reader_list, n_boot: int, seed: int = 42):
    """Combined inter-reader agreement table (patient-level + nodule-level)."""
    rows = []

    # --- Part A: patient-level agreement (categorized pLoM risk, recommendation) ---
    # Fresh RNG (matches agreement_paper.build_table4_alt seed/order) so the CIs reproduce.
    rng = np.random.default_rng(seed)
    sess = {s: build_reader_session_table(exports_dir, reader_list, s) for s in (0, 1)}
    risk = {s: _plom_to_bin(_patient_matrix(sess[s], "patient_lom", reader_list)) for s in (0, 1)}
    fu = {}
    for s in (0, 1):
        raw = _patient_matrix(sess[s], "patient_follow_up_number", reader_list)
        grp = np.full_like(raw, np.nan, dtype=float)
        for src, dst in FOLLOWUP_GROUP_MAP.items():
            grp[raw == src] = dst
        fu[s] = grp

    rows.append(_header("Malignancy risk category"))
    for label, lvl in PLOM_BINS:
        rows.append(_kappa_row(f"  {label}", _binarize(risk[0], lvl), _binarize(risk[1], lvl),
                               [0, 1], n_boot, rng))
    rows.append(_kappa_row("  Overall", risk[0], risk[1], [0, 1, 2, 3], n_boot, rng))

    rows.append(_header("Clinical recommendation"))
    for label, grp_val in FU_GROUPS:
        rows.append(_kappa_row(f"  {label}", _binarize(fu[0], grp_val), _binarize(fu[1], grp_val),
                               [0, 1], n_boot, rng))
    rows.append(_kappa_row("  Overall", fu[0], fu[1], [0, 1, 2], n_boot, rng))

    # --- Part B: nodule-level agreement (detection, size, attenuation) ---
    # Fresh RNG (matches inter_reader_fleiss_kappa_table.py seed/order).
    rng = np.random.default_rng(seed)
    matched = pd.read_csv(matched_path, low_memory=False)
    df = matched[(matched["reader"].isin(reader_list))
                 & (matched["diam_multiplier"] == TABLE4_DM)].copy()
    annotations = build_reader_annotations(exports_dir, reader_list)
    df = df.merge(annotations, on=["key", "reader", "is_aided_encord", "det_region_name"], how="left")

    rows.append(_header("Nodule detection"))
    rows.append(_kappa_row("  Overall",
                           _detection_matrix(df[df.is_aided_encord == 0], reader_list),
                           _detection_matrix(df[df.is_aided_encord == 1], reader_list),
                           [0, 1], n_boot, rng))

    tp = df[df["type"] == "tp"].copy()
    tp["size_category"] = _size_category(tp["det_diameter"])
    tp_u, tp_a = tp[tp.is_aided_encord == 0], tp[tp.is_aided_encord == 1]

    rows.append(_header("Nodule size"))
    size_u = _category_matrix(tp_u, "size_category", reader_list)
    size_a = _category_matrix(tp_a, "size_category", reader_list)
    for cat in SIZE_CATEGORIES:
        rows.append(_kappa_row(f"  {SIZE_DISPLAY[cat]}", _binarize(size_u, cat),
                               _binarize(size_a, cat), [0, 1], n_boot, rng))
    rows.append(_kappa_row("  Overall", size_u, size_a, SIZE_CATEGORIES, n_boot, rng))

    rows.append(_header("Nodule attenuation"))
    att_u = _category_matrix(tp_u, "reader_attenuation", reader_list)
    att_a = _category_matrix(tp_a, "reader_attenuation", reader_list)
    for cat in ATTEN_CATEGORIES:
        rows.append(_kappa_row(f"  {ATTEN_DISPLAY[cat]}", _binarize(att_u, cat),
                               _binarize(att_a, cat), [0, 1], n_boot, rng))
    rows.append(_kappa_row("  Overall", att_u, att_a, ATTEN_CATEGORIES, n_boot, rng))

    return pd.DataFrame(rows, columns=["Variable", "W/o CAD", "With CAD", "p-value"])


TABLE4_CAPTION = (
    "Table 4. Inter-observer agreement (Fleiss' kappa, 95% CI) with and without CAD, for "
    "malignancy-risk category, clinical recommendation, nodule detection, nodule size and "
    "nodule attenuation. Risk category is derived from the categorized likelihood of "
    "malignancy (pLoM: <10, 10-35, 36-70, >=71%); recommendation groups the 9 follow-up "
    "options into No action / CT surveillance / Diagnostic procedure. Risk and recommendation "
    "are patient-level; detection, size and attenuation are nodule-level on matched detections "
    "at a 1x match radius (size and attenuation on true positives). Per-category rows use "
    "one-vs-rest recoding; Overall rows use the full multi-category matrix. 95% CIs are "
    "percentile bootstrap (subjects resampled); p-values are a two-sided bootstrap test on the "
    "paired aided-unaided kappa difference."
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
        if p == "" or pd.isna(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"

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
    rows.append(row("No. of CT scans",
                    {"tot": SD["tot"], "neg": SD["neg"], "nd": SD["nd"], "can": SD["can"], "ben": SD["ben"]},
                    {c: SD["tot"] for c in order}))
    rows.append(row("No. of nodules",
                    {"tot": ND["tot"], "neg": 0, "nd": ND["nd"], "can": ND["can"], "ben": ND["ben"]},
                    {c: ND["tot"] for c in order}))

    rows.append(section("Nodule size (long-axis diameter)", p_size))
    for label, key in [("  Small (<9 mm)", "small"), ("  Medium (9-14 mm)", "medium"),
                       ("  Large (>=15 mm)", "large")]:
        rows.append(row(label, nod_counts(nod.size_bucket, key), ND))

    rows.append(section("Nodule attenuation", p_att))
    for label, key in [("  Solid", "solid"), ("  Part-solid", "part-solid"), ("  GGO", "ggo")]:
        rows.append(row(label, nod_counts(nod.attenuation, key), ND))

    rows.append(section("Nodule margin", p_marg))
    for label, key in [("  Non-spiculated", "no"), ("  Spiculated", "yes")]:
        rows.append(row(label, nod_counts(nod.is_spiculated, key), ND))

    rows.append(section("Level of difficulty (per CT scan)", p_diff))
    for label, key in [("  Easy", "easy"), ("  Hard", "hard")]:
        counts = {"tot": (scan.eh == key).sum(), "neg": (scan_neg.eh == key).sum(),
                  "nd": (scan_pos.eh == key).sum(), "can": (scan_can.eh == key).sum(),
                  "ben": (scan_ben.eh == key).sum()}
        rows.append(row(label, counts, SD))

    cols = ["Characteristic", f"Total (n={SD['tot']})",
            f"Negative - no nodule (n={SD['neg']})", f"Nodule data (n={SD['nd']})",
            f"Cancer (n={SD['can']})", f"Benign (n={SD['ben']})", "p-value (Cancer vs Benign)"]
    return pd.DataFrame(rows, columns=cols)


TABLE_E1_CAPTION = (
    "Table E1. Ground-truth counts for the main-study dataset (one row per annotated "
    "nodule plus one row per negative scan). Cells are n (column %). 'Negative - no "
    "nodule' scans are kept separate from Benign. p-values test Cancer vs Benign per "
    "characteristic: Pearson chi-square (size, margin, difficulty); Fisher-Freeman-Halton "
    "exact for attenuation (small expected counts). Size, attenuation and margin compare "
    "nodules; difficulty compares CT scans."
)


# =========================================================================== #
# Table E4 -- nodule-detection performance (tabular companion of Figure 6)
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
    "Table E4. Nodule-detection performance at a 1x nodule-diameter match radius: "
    "across-reader mean sensitivity and false positives (FP) per case (standard error), "
    "unaided vs aided vs CAD alone, for all nodules and expert-annotated (EA) nodules "
    ">=5 mm. The >=5 mm size filter applies to TP/FN only (standard FROC convention). "
    "This is the tabular companion of Figure 6."
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


if __name__ == "__main__":
    main()
