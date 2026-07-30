"""Secondary endpoint figures for the CLEAREST main study manuscript (Radiology).

This script reproduces the nodule-DETECTION secondary-endpoint figures of the
CLEAREST multi-reader multi-case (MRMC) study, comparing readers unaided
("Without CAD") versus aided by the Lung Cancer Prediction (LCP) AI ("With CAD"),
with the standalone AI ("CAD alone") as a reference.

Figures produced
----------------
Figure 6  -- Free-response ROC (FROC) operating point for nodule detection:
             across-reader mean sensitivity (detection rate) versus mean false
             positives (FP) per case, at a 1x nodule-diameter match radius, for
             unaided readers, aided readers, and CAD alone. Shown for all nodules
             and for expert-annotated (EA) nodules >= 5 mm. Reader points carry
             standard-error bars across readers; CAD alone is a single operating
             point.
Figure E2 -- Bland-Altman agreement of reader vs expert-annotated (EA) long-axis
             nodule diameter on true-positive detections (supplemental material).
             Three panels: unaided readers, aided readers, and AI only. Each panel
             shows the bias (mean signed difference) and 95% limits of agreement
             (bias +/- 1.96 SD).

Both figures use a 1x nodule-diameter match radius to pair detections with the
expert-annotated ground truth.

Data inputs (not distributed with this repository)
--------------------------------------------------
Under ``<data-root>/detection_rate/`` (default data root: ``../resources/clearest_main_study``):
  matched_all_readers_sessions_notn_all_ignore_filtered.csv   (Figure 6)
      Matched detections per (reader, session), with false positives on the
      cardiothoracic-consensus missed-GT ignore-list removed.
  matched_all_readers_sessions_notn_all.csv                   (Figure E2)
      Matched detections per (reader, session), including paired detection and
      ground-truth diameters.

Both matched tables share the columns used here: ``reader`` (the 15 reader ids
plus ``ai`` for the standalone AI), ``is_aided_encord`` (0 = unaided, 1 = aided),
``diam_multiplier`` (match radius as a multiple of nodule diameter), ``type``
(``tp``/``fn``/``fp``), ``key`` (case id), ``gt_diameter`` and ``det_diameter``
(expert-annotated and detected long-axis diameters, mm).

Usage
-----
    python secondary_endpoint_figures.py \
        --data-root /path/to/resources/clearest_main_study \
        --out-dir ./figures --show

The data root may also be supplied via the CLEAREST_DATA_ROOT environment variable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# The 15 readers who completed the study (the "ai" and "average" rows are handled
# separately). Indices 12, 14 and 16 are absent by design.
READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]

OPERATING_DM = 1       # operating point: match radius = 1x nodule diameter
MIN_DIAMETER_MM = 5    # EA-diameter threshold for the >= 5 mm subset (Figure 6)

FONTSIZE = 14


# --------------------------------------------------------------------------- #
# Figure 6 -- detection-rate FROC operating point
# --------------------------------------------------------------------------- #
def cases_per_reader(matched: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct cases read by each (reader, session)."""
    return (matched.groupby(["reader", "is_aided_encord"])["key"]
            .nunique().rename("n_cases").reset_index())


def detection_perf(matched_op: pd.DataFrame, n_cases: pd.DataFrame,
                   gt_min_diam: float | None = None) -> pd.DataFrame:
    """Per-(reader, session) sensitivity and FP-per-case at the operating point.

    Sensitivity = TP / (TP + FN); the optional GT-diameter filter restricts only
    TP/FN (standard FROC convention), leaving FPs and the case count unchanged.
    """
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
    out = out.merge(n_cases, on=["reader", "is_aided_encord"], how="left")
    out["fp_per_case"] = out["fp"] / out["n_cases"]
    return out


def _froc_summary(sub: pd.DataFrame):
    """Across-reader mean and standard error of sensitivity and FP/case."""
    n = len(sub)
    return {
        "sens": sub["sensitivity"].mean(),
        "sens_se": sub["sensitivity"].std(ddof=1) / np.sqrt(n),
        "fpc": sub["fp_per_case"].mean(),
        "fpc_se": sub["fp_per_case"].std(ddof=1) / np.sqrt(n),
        "n": n,
    }


def make_figure6(matched: pd.DataFrame, out_path: Path) -> None:
    """Figure 6 -- reader sensitivity vs FP/case at a 1x-diameter match radius."""
    n_cases = cases_per_reader(matched)
    matched_op = matched[matched["diam_multiplier"] == OPERATING_DM]

    # Two ground-truth subsets: all nodules (black), and EA diameter >= 5 mm (red).
    series = [("All nodules", None, "black"),
              (f"EA >= {MIN_DIAMETER_MM} mm", MIN_DIAMETER_MM, "tab:red")]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, gt_min, color in series:
        perf = detection_perf(matched_op, n_cases, gt_min_diam=gt_min)
        readers = perf[perf["reader"].isin(READERS)]
        ai = perf[perf["reader"] == "ai"]

        unaided = _froc_summary(readers[readers["is_aided_encord"] == 0])
        aided = _froc_summary(readers[readers["is_aided_encord"] == 1])

        print(f"  {label}:")
        print(f"    Unaided (n={unaided['n']}): sens={unaided['sens']:.3f}+/-{unaided['sens_se']:.3f} "
              f"FP/case={unaided['fpc']:.3f}+/-{unaided['fpc_se']:.3f}")
        print(f"    Aided   (n={aided['n']}): sens={aided['sens']:.3f}+/-{aided['sens_se']:.3f} "
              f"FP/case={aided['fpc']:.3f}+/-{aided['fpc_se']:.3f}")

        # CAD alone (single operating point) is drawn only for the >= 5 mm subset.
        if color == "tab:red" and len(ai):
            cad_sens = ai["sensitivity"].iloc[0]
            cad_fpc = ai["fp_per_case"].iloc[0]
            print(f"    CAD alone:         sens={cad_sens:.3f} FP/case={cad_fpc:.3f}")
            ax.scatter([cad_fpc], [cad_sens], marker="D", s=110,
                       facecolors=color, edgecolors=color, zorder=4,
                       label=f"CAD ({label})")

        ax.errorbar(unaided["fpc"], unaided["sens"],
                    xerr=unaided["fpc_se"], yerr=unaided["sens_se"],
                    fmt="o", color=color, markersize=10, capsize=4,
                    markerfacecolor="none", label=f"Unaided ({label})")
        ax.errorbar(aided["fpc"], aided["sens"],
                    xerr=aided["fpc_se"], yerr=aided["sens_se"],
                    fmt="s", color=color, markersize=10, capsize=4,
                    markerfacecolor=color, label=f"Aided ({label})")

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.125, 2)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2"])
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks(np.arange(0.6, 1.05, 0.1))
    ax.set_xlabel("Average number of false positives (SE)", fontsize=12)
    ax.set_ylabel("Sensitivity (SE)", fontsize=12)
    ax.set_title(
        f"Reader sensitivity vs FPs per case (match radius x{OPERATING_DM} diameter)",
        fontsize=12)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Figure E2 -- Bland-Altman of reader vs EA diameter
# --------------------------------------------------------------------------- #
def paired_diameter(matched: pd.DataFrame, readers: list[str] | None) -> pd.DataFrame:
    """True-positive rows with paired (EA, detected) diameters at the 1x radius.

    ``readers=None`` selects the standalone AI (reader == "ai").
    """
    op = matched[(matched["diam_multiplier"] == OPERATING_DM)
                 & (matched["type"] == "tp")]
    op = op[op["reader"] == "ai"] if readers is None else op[op["reader"].isin(readers)]
    df = op.dropna(subset=["gt_diameter", "det_diameter"]).copy()
    df["mean_diameter"] = (df["det_diameter"] + df["gt_diameter"]) / 2
    df["diff_diameter"] = df["det_diameter"] - df["gt_diameter"]
    return df


def _ba_stats(diffs: np.ndarray) -> tuple[float, float, float]:
    """Return (bias, lower 95% LoA, upper 95% LoA)."""
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def _ba_axes_limits(paired: pd.DataFrame):
    x_min = float(paired["mean_diameter"].min())
    x_max = float(paired["mean_diameter"].max())
    y_abs = float(np.percentile(np.abs(paired["diff_diameter"]), 99.5))
    return (max(0, x_min - 1), x_max + 1), (-y_abs * 1.1, y_abs * 1.1)


def _draw_ba_panel(ax, df, panel_label, title, color, x_lim, y_lim):
    """Render one Bland-Altman panel: scatter plus bias and 95% LoA lines."""
    bias, lo, hi = _ba_stats(df["diff_diameter"].to_numpy())
    ax.scatter(df["mean_diameter"], df["diff_diameter"],
               s=14, c=color, alpha=0.35, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(bias, color="black", linewidth=1.2)
    ax.axhline(hi, color="grey", linewidth=1.0, linestyle="--")
    ax.axhline(lo, color="grey", linewidth=1.0, linestyle="--")

    x_text = x_lim[1] - 0.02 * (x_lim[1] - x_lim[0])
    for y, text in [(bias, f"  bias = {bias:+.2f}"),
                    (hi, f"  +1.96 SD = {hi:+.2f}"),
                    (lo, f"  -1.96 SD = {lo:+.2f}")]:
        ax.text(x_text, y, text, fontsize=FONTSIZE - 3,
                va="bottom" if y >= 0 else "top", ha="right", color="black")

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE + 2, fontweight="bold", va="top", ha="left")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def make_figureE2(matched: pd.DataFrame, out_path: Path) -> None:
    """Figure E2 -- Bland-Altman panels: unaided | aided | AI only."""
    reader_paired = paired_diameter(matched, READERS)
    ai_paired = paired_diameter(matched, None)
    x_lim, y_lim = _ba_axes_limits(pd.concat([reader_paired, ai_paired], ignore_index=True))

    panels = [
        (reader_paired[reader_paired["is_aided_encord"] == 0], "A", "Unaided", "tab:blue"),
        (reader_paired[reader_paired["is_aided_encord"] == 1], "B", "Aided", "tab:red"),
        (ai_paired, "C", "AI only", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6), sharex=True, sharey=True)
    for ax, (df, label, session, color) in zip(axes, panels):
        if session == "AI only":
            title = f"{session}  (n={len(df)} TPs, {df['key'].nunique()} studies)"
            ylabel = "AI - EA diameter (mm)"
        else:
            title = (f"{session}  (n={len(df)} TPs across "
                     f"{df['reader'].nunique()} readers, {df['key'].nunique()} studies)")
            ylabel = "Reader - EA diameter (mm)"
        _draw_ba_panel(ax, df, label, title, color, x_lim, y_lim)
        ax.set_xlabel("Mean of detected and EA diameter (mm)", fontsize=FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        bias, lo, hi = _ba_stats(df["diff_diameter"].to_numpy())
        print(f"  {session}: n={len(df)} bias={bias:+.2f} 95% LoA [{lo:+.2f}, {hi:+.2f}]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CLEAREST_DATA_ROOT", "../resources/clearest_main_study")),
        help="Path to the clearest_main_study data directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"),
                        help="Directory to write the figures into.")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures interactively as well as saving them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detection_dir = args.data_root / "detection_rate"

    print("Figure 6 (detection-rate FROC operating point):")
    matched_filtered = pd.read_csv(
        detection_dir / "matched_all_readers_sessions_notn_all_ignore_filtered.csv",
        low_memory=False)
    make_figure6(matched_filtered, args.out_dir / "figure6_detection_rate_froc.png")

    print("Figure E2 (Bland-Altman reader vs EA diameter):")
    matched_all = pd.read_csv(
        detection_dir / "matched_all_readers_sessions_notn_all.csv", low_memory=False)
    make_figureE2(matched_all, args.out_dir / "figureE2_bland_altman_diameter.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
