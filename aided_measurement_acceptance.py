"""Aided nodule measurements accepted unchanged from the device (reviewer R1-5).

In the aided session, each nodule mark carried the question "Accept suggested
Measurements?" (Yes / No / Not_Applicable). When the reader drew no calipers, the export
parser (encord_main/main_study_parse_encord_json.py, preprocess_aided_scans) filled the
mark's long and short axes, solid axes and attenuation from the matched device (LCP AI)
candidate and set ``used_ai_measurements = 1``. A mark with ``used_ai_measurements == 1``
therefore records the device diameter unchanged. ``0`` = the reader's own calipers;
``-1`` = no measurement and no device candidate within 20 mm.

This script reports, for aided reads:
  1. the share of nodule marks whose recorded diameter is the device's, over all aided
     nodule marks and over the true-positive (TP) marks used in Figure E2 panel B
     (matched to an expert-annotated (EA) nodule at 1x the EA long axis), with the range
     across readers; also the stricter share where the reader answered "Yes";
  2. the Bland-Altman bias (reader - EA long axis, mm) and 95% limits of agreement of the
     aided TP marks, split by accepted device value vs reader's own calipers, next to
     the unaided and device-alone (Figure E2 panels A and C) values.

Data inputs (not distributed), under ``<data-root>``:
  final_exports/<reader>_S1_S2_final.csv -- ``study_id``, ``is_aided_encord``, ``is_nodule``,
      ``location_x/y/z_mm``, ``long_diameter``, ``accept_ai_measurements``,
      ``used_ai_measurements``.
  detection_rate/matched_all_readers_sessions_notn_all.csv -- the Figure E2 input:
      ``type``, ``reader``, ``is_aided_encord``, ``diam_multiplier``, ``key``,
      ``det_x/y/z``, ``det_diameter``, ``gt_diameter``.

Output: ``<out-dir>/aided_measurement_acceptance.csv``.

Note: analysis requested by a reviewer (R1-5); findings of this reader study, not a
device performance claim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]
OPERATING_DM = 1


def load_marks(root: Path) -> pd.DataFrame:
    rows = []
    for r in READERS:
        e = pd.read_csv(root / "final_exports" / f"{r}_S1_S2_final.csv")
        e = e[e["is_nodule"] == 1]
        rows.append(pd.DataFrame({
            "reader": r, "is_aided_encord": e["is_aided_encord"], "key": e["study_id"],
            "det_x": e["location_x_mm"], "det_y": e["location_y_mm"], "det_z": e["location_z_mm"],
            "accept": e["accept_ai_measurements"].fillna("missing"),
            "used_ai": e["used_ai_measurements"]}))
    return pd.concat(rows, ignore_index=True)


def tp_marks(root: Path, marks: pd.DataFrame) -> pd.DataFrame:
    """Figure E2 TP rows (readers, 1x radius) with the acceptance flag joined on."""
    m = pd.read_csv(root / "detection_rate" / "matched_all_readers_sessions_notn_all.csv",
                    low_memory=False)
    tp = m[(m["diam_multiplier"] == OPERATING_DM) & (m["type"] == "tp")
           & m["reader"].isin(READERS)].dropna(subset=["gt_diameter", "det_diameter"]).copy()
    tp = tp[["reader", "is_aided_encord", "key", "det_x", "det_y", "det_z",
             "det_diameter", "gt_diameter"]]
    mk = marks.copy()
    for c in ("det_x", "det_y", "det_z"):
        tp[c + "_r"] = tp[c].round(3)
        mk[c + "_r"] = mk[c].round(3)
    keys = ["reader", "is_aided_encord", "key", "det_x_r", "det_y_r", "det_z_r"]
    mk = mk.drop_duplicates(keys)
    out = tp.merge(mk[keys + ["accept", "used_ai"]], on=keys, how="left", validate="m:1",
                   indicator=True)
    if (out["_merge"] != "both").any():
        raise ValueError(f"{(out['_merge'] != 'both').sum()} TP marks not found in the exports")
    out["diff"] = out["det_diameter"] - out["gt_diameter"]
    return out


def _pct(n, d):
    return f"{n} of {d} ({100 * n / d:.1f}%)"


def _ba(diff: pd.Series) -> str:
    b, sd = diff.mean(), diff.std(ddof=1)
    return f"{b:+.2f} ({b - 1.96 * sd:+.2f}, {b + 1.96 * sd:+.2f}); n = {len(diff)}"


def build(root: Path) -> pd.DataFrame:
    marks = load_marks(root)
    tp = tp_marks(root, marks)
    rows = []

    def add(section, item, value):
        rows.append({"Section": section, "Item": item, "Value": value})

    aided = marks[marks["is_aided_encord"] == 1]
    tpa = tp[tp["is_aided_encord"] == 1]
    for lbl, d in [("All aided nodule marks", aided),
                   ("Aided TP marks (Figure E2 panel B)", tpa)]:
        acc = d["used_ai"] == 1
        add("Accepted unchanged", f"{lbl}: recorded diameter = device diameter",
            _pct(int(acc.sum()), len(d)))
        add("Accepted unchanged", f"{lbl}: reader answered 'Yes' and diameter = device",
            _pct(int((acc & (d["accept"] == "Yes")).sum()), len(d)))
        per = d.groupby("reader")["used_ai"].apply(lambda s: 100 * (s == 1).mean())
        add("Accepted unchanged", f"{lbl}: per-reader range (median)",
            f"{per.min():.0f}%-{per.max():.0f}% ({per.median():.0f}%)")
        add("Accepted unchanged", f"{lbl}: reader's own calipers",
            _pct(int((d["used_ai"] == 0).sum()), len(d)))
        add("Accepted unchanged", f"{lbl}: no measurement (no device candidate)",
            _pct(int((d["used_ai"] == -1).sum()), len(d)))

    tpu = tp[tp["is_aided_encord"] == 0]
    add("Bias vs EA, mm (95% LoA)", "Unaided TP marks (Figure E2 A)", _ba(tpu["diff"]))
    add("Bias vs EA, mm (95% LoA)", "Aided TP marks (Figure E2 B)", _ba(tpa["diff"]))
    add("Bias vs EA, mm (95% LoA)", "  aided, device diameter accepted",
        _ba(tpa.loc[tpa["used_ai"] == 1, "diff"]))
    add("Bias vs EA, mm (95% LoA)", "  aided, reader's own calipers",
        _ba(tpa.loc[tpa["used_ai"] == 0, "diff"]))
    m = pd.read_csv(root / "detection_rate" / "matched_all_readers_sessions_notn_all.csv",
                    low_memory=False)
    ai = m[(m["diam_multiplier"] == OPERATING_DM) & (m["type"] == "tp") & (m["reader"] == "ai")]
    ai = ai.dropna(subset=["gt_diameter", "det_diameter"])
    add("Bias vs EA, mm (95% LoA)", "Device alone (Figure E2 C)",
        _ba(ai["det_diameter"] - ai["gt_diameter"]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ.get("CLEAREST_DATA_ROOT",
                                                "../resources/clearest_main_study")))
    ap.add_argument("--out-dir", type=Path, default=Path("tables"))
    args = ap.parse_args()
    table = build(args.data_root)
    with pd.option_context("display.max_colwidth", 100, "display.width", 200):
        print(table.to_string(index=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "aided_measurement_acceptance.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
