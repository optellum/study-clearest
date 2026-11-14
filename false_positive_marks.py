"""Characterization of reader false-positive nodule marks with vs without CAD (reviewer R1-4).

With CAD, readers placed more false-positive (FP) nodule marks (Table E4: 0.30 vs 0.50 per
examination). This script describes those marks. It answers three questions:

1. Size: what share of FP marks measured < 6 mm (below the Fleischner 2017 follow-up
   threshold for a solid nodule in a low-risk patient).
2. Device prompting: what share of FP marks lay on a device (LCP AI) candidate shown in
   the aided read.
3. Management: in how many examinations an additional aided FP mark was accompanied by a
   more intensive recommendation than the same reader gave without CAD.

Definitions
-----------
FP mark
    A reader nodule mark with no expert-annotated (EA) nodule within 1x the EA long-axis
    diameter (the Table E4 / Figure 5 operating point), after removing marks on the
    cardiothoracic-consensus ignore list (likely real nodules missed by the reference).
    This is the FP set behind the Table E4 FP-per-case values.
Size
    The reader's own measurement of the mark: long axis, and mean diameter = (long + short)
    / 2, rounded to 0.1 mm. The Fleischner 6 mm threshold uses mean diameter, so it is the
    main measure; the long axis is also reported. Marks without a short axis use the long axis.
Device candidate
    A detection in ``dataset/main_athena_results_0994_5.csv``, the device output the readers
    could see in the aided session. A mark is "on a device candidate" if its centre lies
    within max(reader long axis, device long axis) of a candidate in the same examination.
    (A mark without a measurement uses the device long axis.) The same test is applied to unaided marks as a reference: the device was not shown in
    the unaided session, so the unaided rate is the overlap expected without prompting.
Additional aided FP mark
    An aided FP mark with no unaided FP mark by the same reader in the same examination
    within max(both long axes). These are the marks that the aided read added.
Management change
    Per (reader, examination) with at least one additional aided FP mark, the 9-level
    recommendation (``patient_follow_up_number`` 0-8, ordered by intensity) without and
    with CAD. "Escalated" = more intensive with CAD. "New surveillance or procedure" = the
    3-group recommendation (Table 4: No action 0-1, CT surveillance 2-4, Diagnostic procedure
    5-8) moved from No action to CT surveillance or Diagnostic procedure. The same rates are
    given for reader-examinations without an additional FP mark as a reference. The
    recommendation is per patient, so an escalation is not proof that the FP mark caused it.

Counts are pooled over 15 readers; percentages are of pooled marks or reader-examinations.

Data inputs (not distributed with this repository), under ``<data-root>``:
  detection_rate/matched_all_readers_sessions_notn_all_ignore_filtered.csv
      Matched detections: ``type`` (tp/fn/fp), ``reader``, ``is_aided_encord``,
      ``diam_multiplier``, ``key`` (stage-1 study id), ``det_x/y/z`` (mm), ``det_diameter``.
  final_exports/<reader>_S1_S2_final.csv
      Reader exports: ``study_id``, ``is_aided_encord``, ``is_nodule``,
      ``location_x/y/z_mm``, ``long_diameter``, ``short_diameter``,
      ``patient_follow_up_number``.
  dataset/main_athena_results_0994_5.csv
      Device candidates: ``study_id``, ``det_x/y/z`` (mm), ``whole_long_axis_diameter``.
  dataset/clearest_patient_data_240_main_study_max_lcp_score_gt.csv
      ``study_id_stage1``, ``is_cancer``.

Output: ``<out-dir>/false_positive_marks.csv``.

Note: analysis requested by a reviewer (R1-4); findings of this reader study, not a
device performance claim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]
OPERATING_DM = 1          # Table E4 match radius (1x EA long-axis diameter)
SIZE_THRESHOLD_MM = 6     # Fleischner 2017 follow-up threshold (mean diameter)
# patient_follow_up_number 0..8 -> No action (0) | CT surveillance (1) | Diagnostic procedure (2)
FOLLOWUP_GROUP_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}


def round_half_up(x, nd=1):
    """Round to ``nd`` decimals, halves up (as the size bins of Tables 2 and E1)."""
    f = 10 ** nd
    return np.floor(np.asarray(x, dtype=float) * f + 0.5) / f


def load_fp_marks(data_root: Path) -> pd.DataFrame:
    """FP marks of the 15 readers at the Table E4 operating point, with reader diameters."""
    m = pd.read_csv(data_root / "detection_rate" /
                    "matched_all_readers_sessions_notn_all_ignore_filtered.csv", low_memory=False)
    fp = m[(m["type"] == "fp") & (m["diam_multiplier"] == OPERATING_DM)
           & m["reader"].isin(READERS)].copy()
    fp = fp[["reader", "is_aided_encord", "key", "det_x", "det_y", "det_z", "det_diameter"]]

    # Join the reader's short axis from the export (same mark: same reader, exam, arm, centre).
    exports = []
    for r in READERS:
        e = pd.read_csv(data_root / "final_exports" / f"{r}_S1_S2_final.csv")
        e = e[e["is_nodule"] == 1]
        exports.append(pd.DataFrame({
            "reader": r, "is_aided_encord": e["is_aided_encord"], "key": e["study_id"],
            "det_x": e["location_x_mm"], "det_y": e["location_y_mm"], "det_z": e["location_z_mm"],
            "long_mm": e["long_diameter"], "short_mm": e["short_diameter"]}))
    ex = pd.concat(exports, ignore_index=True)
    for c in ("det_x", "det_y", "det_z"):
        fp[c + "_r"] = fp[c].round(3)
        ex[c + "_r"] = ex[c].round(3)
    keys = ["reader", "is_aided_encord", "key", "det_x_r", "det_y_r", "det_z_r"]
    ex = ex.drop_duplicates(keys)
    out = fp.merge(ex[keys + ["long_mm", "short_mm"]], on=keys, how="left", validate="m:1",
                   indicator=True)
    if (out["_merge"] != "both").any():
        raise ValueError(f"{(out['_merge'] != 'both').sum()} FP marks not found in the reader exports")
    if not np.allclose(out["long_mm"], out["det_diameter"], equal_nan=True):
        raise ValueError("export long axis disagrees with the matched det_diameter")
    short = out["short_mm"].where(out["short_mm"] > 0, out["long_mm"])
    out["mean_mm"] = round_half_up((out["long_mm"] + short) / 2)
    out["long_mm"] = round_half_up(out["long_mm"])
    out["no_size"] = out["long_mm"].isna()          # mark saved without a measurement
    out["short_missing"] = ~(out["short_mm"] > 0) & ~out["no_size"]
    return out.drop(columns=["det_x_r", "det_y_r", "det_z_r", "_merge"]).reset_index(drop=True)


def _near(marks: pd.DataFrame, others: pd.DataFrame, other_diam: str) -> np.ndarray:
    """True where a mark lies within max(mark long axis, other long axis) of an ``other``
    point in the same examination (``key``)."""
    hit = np.zeros(len(marks), dtype=bool)
    groups = {k: g for k, g in others.groupby("key")}
    for i, row in enumerate(marks.itertuples(index=False)):
        g = groups.get(row.key)
        if g is None:
            continue
        d = np.sqrt((g["det_x"] - row.det_x) ** 2 + (g["det_y"] - row.det_y) ** 2
                    + (g["det_z"] - row.det_z) ** 2)
        hit[i] = bool((d <= np.fmax(row.det_diameter, g[other_diam])).any())
    return hit


def flag_marks(fp: pd.DataFrame, device: pd.DataFrame) -> pd.DataFrame:
    fp = fp.copy()
    fp["lt6_mean"] = fp["mean_mm"] < SIZE_THRESHOLD_MM   # False for marks without a size
    fp["lt6_long"] = fp["long_mm"] < SIZE_THRESHOLD_MM
    fp["on_device"] = _near(fp, device, "dev_diameter")
    fp["additional"] = False
    for r in READERS:
        a = fp.index[(fp["reader"] == r) & (fp["is_aided_encord"] == 1)]
        u = fp[(fp["reader"] == r) & (fp["is_aided_encord"] == 0)]
        fp.loc[a, "additional"] = ~_near(fp.loc[a], u, "det_diameter")
    return fp


def load_recommendations(data_root: Path) -> pd.DataFrame:
    rows = []
    for r in READERS:
        e = pd.read_csv(data_root / "final_exports" / f"{r}_S1_S2_final.csv")
        rec = e.groupby(["study_id", "is_aided_encord"])["patient_follow_up_number"].first().unstack()
        rec = rec.rename(columns={0: "fu_unaided", 1: "fu_aided"}).reset_index()
        rec["reader"] = r
        rows.append(rec.rename(columns={"study_id": "key"}))
    rec = pd.concat(rows, ignore_index=True)
    if rec[["fu_unaided", "fu_aided"]].isna().any().any():
        raise ValueError("missing recommendations")
    rec["escalated"] = rec["fu_aided"] > rec["fu_unaided"]
    g_u = rec["fu_unaided"].map(FOLLOWUP_GROUP_MAP)
    g_a = rec["fu_aided"].map(FOLLOWUP_GROUP_MAP)
    rec["new_surv_or_proc"] = (g_u == 0) & (g_a > 0)
    rec["new_surv"] = (g_u == 0) & (g_a == 1)   # No action -> CT surveillance
    rec["new_proc"] = (g_u == 0) & (g_a == 2)   # No action -> diagnostic procedure
    return rec


def _pct(n, d):
    return f"{n} of {d} ({100 * n / d:.1f}%)" if d else "0 of 0"


def build_table(fp: pd.DataFrame, rec: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    rows = []

    def add(section, item, value):
        rows.append({"Section": section, "Item": item, "Value": value})

    una, aid = fp[fp["is_aided_encord"] == 0], fp[fp["is_aided_encord"] == 1]
    add_ = aid[aid["additional"]]
    add("FP marks", "Unaided FP marks (15 readers, 240 examinations)", str(len(una)))
    add("FP marks", "Aided FP marks", str(len(aid)))
    add("FP marks", "Aided FP marks without an unaided FP mark at the same site (additional)",
        str(len(add_)))
    for lbl, d in [("Unaided FP marks", una), ("Aided FP marks", aid),
                   ("Additional aided FP marks", add_)]:
        n = int((~d["no_size"]).sum())   # denominator: marks with a measurement
        add("Size < 6 mm", f"{lbl}, mean diameter < 6 mm", _pct(int(d["lt6_mean"].sum()), n))
        add("Size < 6 mm", f"{lbl}, long axis < 6 mm", _pct(int(d["lt6_long"].sum()), n))
        add("Size", f"{lbl}, mean diameter median (IQR), mm",
            f"{d['mean_mm'].median():.1f} ({d['mean_mm'].quantile(.25):.1f}, "
            f"{d['mean_mm'].quantile(.75):.1f})")
    add("Size", "FP marks without a short axis (long axis used for the mean)",
        str(int(fp["short_missing"].sum())))
    add("Size", "FP marks without any measurement (excluded from size rows)",
        str(int(fp["no_size"].sum())))
    for lbl, d in [("Unaided FP marks (reference; device not shown)", una),
                   ("Aided FP marks", aid), ("Additional aided FP marks", add_)]:
        add("On a device candidate", lbl, _pct(int(d["on_device"].sum()), len(d)))

    # Management, per (reader, examination).
    re_ = rec.copy()
    with_add = add_.groupby(["reader", "key"]).size().rename("n_add").reset_index()
    re_ = re_.merge(with_add, on=["reader", "key"], how="left").fillna({"n_add": 0})
    re_["is_cancer"] = re_["key"].map(truth)
    if re_["is_cancer"].isna().any():
        raise ValueError("examinations without ground truth")
    dev_add = add_[add_["on_device"]].groupby(["reader", "key"]).size().rename("n_add_dev")
    re_ = re_.merge(dev_add.reset_index(), on=["reader", "key"], how="left").fillna({"n_add_dev": 0})
    groups = [
        ("Reader-examinations with >=1 additional aided FP mark", re_["n_add"] > 0),
        ("  of which non-malignant", (re_["n_add"] > 0) & (re_["is_cancer"] == 0)),
        ("  of which malignant", (re_["n_add"] > 0) & (re_["is_cancer"] == 1)),
        ("  with >=1 additional mark on a device candidate, non-malignant",
         (re_["n_add_dev"] > 0) & (re_["is_cancer"] == 0)),
        ("Reference: reader-examinations without an additional FP mark, non-malignant",
         (re_["n_add"] == 0) & (re_["is_cancer"] == 0)),
    ]
    # Mark level (the letter's "Z%"): additional aided FP marks in reader-examinations
    # whose recommendation moved from No action to surveillance or a procedure with CAD.
    mk = add_.merge(re_[["reader", "key", "new_surv_or_proc", "escalated", "is_cancer"]],
                    on=["reader", "key"], how="left", validate="m:1")
    add("Management (marks)", "Additional aided FP marks in a reader-examination escalated with CAD "
        "(9-level)", _pct(int(mk["escalated"].sum()), len(mk)))
    add("Management (marks)", "Additional aided FP marks in a reader-examination moved from No action "
        "to CT surveillance or diagnostic procedure", _pct(int(mk["new_surv_or_proc"].sum()), len(mk)))
    nm = mk[mk["is_cancer"] == 0]
    add("Management (marks)", "  same, non-malignant examinations only",
        _pct(int(nm["new_surv_or_proc"].sum()), len(nm)))
    for lbl, mask in groups:
        d = re_[mask]
        add("Management", f"{lbl}: n", str(len(d)))
        add("Management", f"{lbl}: recommendation more intensive with CAD (9-level)",
            _pct(int(d["escalated"].sum()), len(d)))
        add("Management", f"{lbl}: No action -> CT surveillance or diagnostic procedure",
            _pct(int(d["new_surv_or_proc"].sum()), len(d)))
        add("Management", f"{lbl}:   No action -> CT surveillance",
            _pct(int(d["new_surv"].sum()), len(d)))
        add("Management", f"{lbl}:   No action -> diagnostic procedure",
            _pct(int(d["new_proc"].sum()), len(d)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ.get("CLEAREST_DATA_ROOT",
                                                "../resources/clearest_main_study")))
    ap.add_argument("--out-dir", type=Path, default=Path("tables"))
    args = ap.parse_args()
    root = args.data_root

    device = pd.read_csv(root / "dataset" / "main_athena_results_0994_5.csv")
    device = device.rename(columns={"study_id": "key",
                                    "whole_long_axis_diameter": "dev_diameter"})
    device = device[["key", "det_x", "det_y", "det_z", "dev_diameter"]]
    truth = pd.read_csv(root / "dataset" / "clearest_patient_data_240_main_study_max_lcp_score_gt.csv",
                        usecols=["study_id_stage1", "is_cancer"]).set_index("study_id_stage1")["is_cancer"]

    fp = flag_marks(load_fp_marks(root), device)
    table = build_table(fp, load_recommendations(root), truth)
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(table.to_string(index=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "false_positive_marks.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
