"""Primary endpoint figures for the CLEAREST main study manuscript (Radiology).

This script reproduces the primary-endpoint figures of the CLEAREST multi-reader
multi-case (MRMC) study. The endpoint is the change in reader performance,
measured by the area under the ROC curve (AUC) of the patient-level likelihood of
malignancy (``patient_lom``), when reading is aided by the Lung Cancer Prediction
(LCP) AI ("With CAD") versus unaided ("Without CAD"). The standalone AI score is
shown as a "CAD alone" reference.

Figures produced
----------------
Figure 4  -- Average reader ROC curves: Without CAD, With CAD, and CAD alone. The
             reader-arm AUC and 95% CI shown are the DBM (Wilcoxon FOM, RRRC) values
             from effect_size_cohort.R; CAD alone is an empirical AUC (bootstrap CI).
Figure 5  -- Effect size by reader sub-specialty. Mean unaided AUC, mean aided AUC,
             and their difference (the AI effect size) for cardiothoracic
             specialists, other radiologists, and all readers combined.
Figure E1 -- Effect size for every reader (supplemental material). Per-reader
             unaided AUC, aided AUC, and their difference, with an asterisk marking
             readers whose aided-vs-unaided difference is statistically significant
             (p < 0.05, DeLong test).

All figures are computed over all patients and are based on the patient-level
likelihood of malignancy. Per-reader significance (Figure E1 asterisks) comes from
the DeLong test in ``MAIN_delong.R``.

Study design and per-reader AUC
-------------------------------
The study uses a paired crossover design: each reader reads every patient once
unaided and once aided, across two stages (S1 and S2). Reader exports use a
stage-specific ``study_id``; the S2 identifiers are remapped to their stage-1
counterpart so that ground truth can be attached and each patient contributes
exactly one aided and one unaided read. A reader's unaided (resp. aided) AUC is
therefore computed over all 240 patients as read in the unaided (resp. aided) arm.

Data inputs (not distributed with this repository)
--------------------------------------------------
Under ``<data-root>`` (default: ``../resources/clearest_main_study``):
  dataset/clearest_patient_data_240_main_study_max_lcp_score.csv
      Patient-level table for the 240-patient cohort. Provides ground truth
      (``is_cancer``), the stage-specific study identifiers (``study_id_stage1``,
      ``study_id_stage2``), and the standalone LCP AI score
      (``zclcp_lcp_score_max_y``) used for the "CAD alone" curve.
  dataset/participant_demographic.csv
      Reader demographics; ``sub_specialty`` is used to assign the specialty group.
  final_exports/<reader>_S1.csv, final_exports/<reader>_S2.csv
      Per-reader annotation exports for stages S1 and S2. Provide the reader's
      ``patient_lom`` score, the aided/unaided flag (``is_aided_encord``), the read
      ``stage``, and the per-stage ``study_id``.
  mrmc/cohort_auc_by_treatment.csv   (Figure 4 reader-arm AUC/CI)
      Per-treatment DBM AUC and RRRC 95% CI, written by effect_size_cohort.R.
      Run that R script before this one.

Usage
-----
    python figures5_E1_primary_endpoint.py \
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
from sklearn.metrics import auc, roc_auc_score, roc_curve

import matplotlib.pyplot as plt


# Readers who completed the study, in display order. Reader indices 12, 14 and 16
# are absent by design (those participant slots were not completed).
READERS = ["MP0", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "MP7", "MP8",
           "MP9", "MP10", "MP11", "MP13", "MP15", "MP17"]

# Per-reader p-values for the aided-vs-unaided AUC difference, from the DeLong
# test in MAIN_delong.R. Used only to annotate significant readers in Figure E1.
READER_P_VALUE = {
    "MP0": 0.243, "MP1": 0.001, "MP2": 0.003, "MP3": 0.251, "MP4": 0.001,
    "MP5": 0.027, "MP6": 0.004, "MP7": 0.001, "MP8": 0.001, "MP9": 0.021,
    "MP10": 0.382, "MP11": 0.015, "MP13": 0.008, "MP15": 0.001, "MP17": 0.009,
}

# Score columns: the reader's malignancy score and the standalone AI score.
READER_SCORE_COL = "patient_lom"          # patient-level likelihood of malignancy (0-100)
AI_SCORE_COL = "zclcp_lcp_score_max_y"     # standalone LCP AI score, "CAD alone"

N_BOOTSTRAPS = 2000  # bootstrap iterations for the AUC confidence intervals
BOOTSTRAP_SEED = 42  # fixed seed for reproducible bootstrap CIs

# Stacked bar colors (Figures 5 and E1).
COLOR_UNAIDED = "#d3d3d3"      # "Without CAD" base bar
COLOR_IMPROVEMENT = "#5a5a5a"  # aided-minus-unaided improvement, stacked on top

# ROC curve colors (Figure 4).
COLOR_ROC_UNAIDED = "#1f77b4"
COLOR_ROC_AIDED = "#2ca02c"
COLOR_ROC_AI = "#d62728"

# Common false-positive-rate grid onto which per-reader ROC curves are interpolated.
MEAN_FPR = np.linspace(0, 1, 101)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_patient_table(data_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        data_root / "dataset" / "clearest_patient_data_240_main_study.csv") # ct-scan cases metadata.


def load_reader_frames(data_root: Path, patients: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load and preprocess each reader's reads into one row per (patient, arm).

    S2 ``study_id`` values are remapped to their stage-1 counterpart, ground truth
    (``is_cancer``) is attached, and the frame is reduced to one row per read
    (unique ``filename``). The remap lets a single lookup key both stages, so each
    patient appears once in the unaided arm and once in the aided arm.
    """
    exports_dir = data_root / "final_exports"
    stage2_to_stage1 = dict(zip(patients["study_id_stage2"], patients["study_id_stage1"]))
    truth = dict(zip(patients["study_id_stage1"], patients["is_cancer"]))

    frames = {}
    for reader in READERS:
        s1 = pd.read_csv(exports_dir / f"{reader}_S1.csv")
        s2 = pd.read_csv(exports_dir / f"{reader}_S2.csv")
        df = pd.concat([s1, s2])
        df.loc[df["stage"] == "S2", "study_id"] = (
            df.loc[df["stage"] == "S2", "study_id"].map(stage2_to_stage1))
        df["is_cancer"] = df["study_id"].map(truth)
        # patient_lom is patient-level, so any nodule row carries the same score;
        # keep one row per read (filename is stage-tagged, so both arms survive).
        df = df.groupby("filename").first().reset_index()
        df = df.dropna(subset=["is_cancer", READER_SCORE_COL])
        frames[reader] = df
    return frames


# --------------------------------------------------------------------------- #
# AUC computation
# --------------------------------------------------------------------------- #
def reader_arm_auc(frame: pd.DataFrame, is_aided: int) -> float:
    """AUC (%) of patient_lom for one reader's aided (1) or unaided (0) reads."""
    arm = frame[frame["is_aided_encord"] == is_aided]
    return roc_auc_score(arm["is_cancer"], arm[READER_SCORE_COL]) * 100.0


def load_dbm_auc(data_root: Path) -> dict:
    """Per-treatment DBM AUC and RRRC 95% CI (fractions), keyed by arm.

    Read from ``mrmc/cohort_auc_by_treatment.csv``, written by
    ``effect_size_cohort.R`` (run it first). Returns
    ``{"unaided": (auc, lo, hi), "aided": (auc, lo, hi)}``.
    """
    path = data_root / "mrmc" / "cohort_auc_by_treatment.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run effect_size_cohort.R first to produce the "
            "DBM per-treatment AUC/CI shown on Figure 4.")
    df = pd.read_csv(path).set_index("arm")
    return {arm: (df.loc[arm, "auc"], df.loc[arm, "ci_lower"], df.loc[arm, "ci_upper"])
            for arm in ("unaided", "aided")}


def cad_alone_curve(patients: pd.DataFrame):
    """ROC curve, AUC and bootstrap CI for the standalone LCP AI ("CAD alone")."""
    ai = patients.dropna(subset=[AI_SCORE_COL, "is_cancer"])
    labels = ai["is_cancer"].to_numpy()
    scores = ai[AI_SCORE_COL].to_numpy()
    fpr, tpr, _ = roc_curve(labels, scores)
    _, lo, hi = bootstrap_auc_ci(labels, scores)
    return fpr, tpr, auc(fpr, tpr), lo, hi


def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=N_BOOTSTRAPS, ci=0.95):
    """Bootstrap mean and CI for a single AUC by resampling cases with replacement."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_scores[idx]))
    scores = np.sort(scores)
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi


def mean_reader_roc(frames, is_aided):
    """Across-reader mean TPR on the common FPR grid for one reading arm."""
    tprs = []
    for f in frames.values():
        arm = f[f["is_aided_encord"] == is_aided]
        fpr, tpr, _ = roc_curve(arm["is_cancer"], arm[READER_SCORE_COL])
        interp = np.interp(MEAN_FPR, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_tpr


def build_reader_table(frames, data_root: Path) -> pd.DataFrame:
    """Per-reader unaided/aided AUC, effect size, and specialty group."""
    demographics = pd.read_csv(data_root / "dataset" / "participant_demographic.csv")
    cardiothoracic = set(
        demographics.loc[
            demographics["sub_specialty"].str.contains("Cardiothoracic", na=False),
            "participant_id",
        ]
    )
    rows = []
    for reader in READERS:
        unaided = reader_arm_auc(frames[reader], is_aided=0)
        aided = reader_arm_auc(frames[reader], is_aided=1)
        rows.append({
            "reader": reader,
            "unaided": unaided,
            "aided": aided,
            "effect_size": aided - unaided,
            "group": "Cardiothoracic" if reader in cardiothoracic else "Other",
            "p_value": READER_P_VALUE[reader],
        })
    return pd.DataFrame(rows)


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and normal-approximation confidence interval across readers."""
    a = np.asarray(values, dtype=float)
    mean = a.mean()
    half = 1.96 * a.std() / np.sqrt(len(a))  # 1.96 -> ~95% CI
    return mean, mean - half, mean + half


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figure4(frames, patients: pd.DataFrame, dbm_auc: dict, out_path: Path) -> None:
    """Figure 4 -- average reader ROC curves (Without CAD, With CAD, CAD alone).

    The reader-arm ROC curves are the across-reader mean TPR, but their AUC and 95%
    CI are the DBM (Wilcoxon FOM) reader-averaged AUC with the random-reader
    random-case CI, taken from ``effect_size_cohort.R`` (via ``load_dbm_auc``) so
    they match the manuscript. CAD alone is the standalone AI: an empirical AUC with
    a case-bootstrap CI (a single rater is not a DBM treatment).
    """
    mean_tpr_unaided = mean_reader_roc(frames, is_aided=0)
    mean_tpr_aided = mean_reader_roc(frames, is_aided=1)
    mean_auc_u, lo_u, hi_u = dbm_auc["unaided"]
    mean_auc_a, lo_a, hi_a = dbm_auc["aided"]
    fpr_ai, tpr_ai, auc_ai, lo_ai, hi_ai = cad_alone_curve(patients)
    # AUCs are displayed as percentages with one decimal (e.g. 88.5).
    print(f"  Without CAD AUC {mean_auc_u * 100:.1f} % ({lo_u * 100:.1f}, {hi_u * 100:.1f})")
    print(f"  With CAD    AUC {mean_auc_a * 100:.1f} % ({lo_a * 100:.1f}, {hi_a * 100:.1f})")
    print(f"  CAD alone   AUC {auc_ai * 100:.1f} % ({lo_ai * 100:.1f}, {hi_ai * 100:.1f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_ai, tpr_ai, color=COLOR_ROC_AI, lw=2, linestyle="dashed",
            label=f"CAD alone: AUC = {auc_ai * 100:.1f} (95% CI: {lo_ai * 100:.1f}, {hi_ai * 100:.1f})")
    ax.plot(MEAN_FPR, mean_tpr_unaided, color=COLOR_ROC_UNAIDED, lw=2,
            label=f"Without CAD: AUC = {mean_auc_u * 100:.1f} (95% CI: {lo_u * 100:.1f}, {hi_u * 100:.1f})")
    ax.plot(MEAN_FPR, mean_tpr_aided, color=COLOR_ROC_AIDED, lw=2,
            label=f"With CAD: AUC = {mean_auc_a * 100:.1f} (95% CI: {lo_a * 100:.1f}, {hi_a * 100:.1f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False-Positive Rate", fontsize=13)
    ax.set_ylabel("True-Positive Rate", fontsize=13)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {out_path}")


def make_figure5(reader_df: pd.DataFrame, cad_auc: float, out_path: Path) -> None:
    """Figure 5 -- effect size by reader sub-specialty."""
    # Group order shown on the x-axis: specialists, other radiologists, everyone.
    groups = ["Cardiothoracic", "Other", "All"]
    labels = {
        "Cardiothoracic": f"Cardiothoracic\n(n={(reader_df['group'] == 'Cardiothoracic').sum()})",
        "Other": f"Other\n(n={(reader_df['group'] == 'Other').sum()})",
        "All": f"All\n(n={len(reader_df)})",
    }

    unaided, aided = [], []
    for group in groups:
        sub = reader_df if group == "All" else reader_df[reader_df["group"] == group]
        unaided.append(sub["unaided"].mean())
        aided.append(sub["aided"].mean())
        # 95% CIs across readers are reported to the console (not drawn on the bars).
        print(f"  {group:<15} unaided {mean_ci(sub['unaided'])[0]:.1f} "
              f"aided {mean_ci(sub['aided'])[0]:.1f} "
              f"effect {sub['aided'].mean() - sub['unaided'].mean():+.1f}")
    unaided = np.array(unaided)
    improvement = np.array(aided) - unaided

    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.arange(len(groups))
    bar_width = 0.4

    ax.bar(x, unaided, width=bar_width, color=COLOR_UNAIDED,
           label="Without CAD", edgecolor="none")
    ax.bar(x, improvement, width=bar_width, bottom=unaided, color=COLOR_IMPROVEMENT,
           label="Improvement with CAD", edgecolor="none")

    # Reference line at the standalone AI ("CAD alone") AUC.
    ax.axhline(y=cad_auc, color="black", linestyle="--", linewidth=1,
               label=f"CAD alone ({cad_auc:.0f}%)")

    fontsize = 14
    ax.set_ylabel("Average AUC (%)", fontsize=fontsize)
    ax.set_ylim(50, 100)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.legend(frameon=False, fontsize=fontsize, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels: unaided AUC below the grey bar, aided AUC above the stack.
    for xi, base, total in zip(x, unaided, aided):
        ax.text(xi, base - 1, f"{base:.1f}", ha="center", va="top",
                fontsize=fontsize, color="black")
        ax.text(xi, total + 1, f"{total:.1f}", ha="center", va="bottom",
                fontsize=fontsize, color="black")

    ax.set_xticks([])
    ax.xaxis.set_ticks_position("none")
    for xi, group in zip(x, groups):
        ax.text(xi, 49, labels[group], ha="center", va="top",
                fontsize=fontsize, color="black", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {out_path}")


def make_figureE1(reader_df: pd.DataFrame, out_path: Path) -> None:
    """Figure E1 -- effect size for every reader (supplemental)."""
    unaided = reader_df["unaided"].to_numpy()
    aided = reader_df["aided"].to_numpy()
    improvement = aided - unaided
    x = np.arange(1, len(reader_df) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.7

    ax.bar(x, unaided, width=bar_width, color=COLOR_UNAIDED,
           label="Without CAD", edgecolor="none")
    ax.bar(x, improvement, width=bar_width, bottom=unaided, color=COLOR_IMPROVEMENT,
           label="Improvement with CAD", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xlabel("Reader", fontsize=12, labelpad=8)
    ax.set_ylabel("AUC (%)", fontsize=12)
    ax.set_ylim(50, 100)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.legend(
        frameon=False, fontsize=11, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.1),
        handles=[
            plt.Line2D([0], [0], color=COLOR_UNAIDED, lw=4, label="Without CAD"),
            plt.Line2D([0], [0], color=COLOR_IMPROVEMENT, lw=4, label="Improvement with CAD"),
        ],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels and a significance asterisk (DeLong p < 0.05) per reader.
    for xi, base, total, p in zip(x, unaided, aided, reader_df["p_value"]):
        ax.text(xi, base - 1, f"{base:.0f}", ha="center", va="top",
                fontsize=9, color="black")
        ax.text(xi, total + 1, f"{total:.0f}", ha="center", va="bottom",
                fontsize=9, color="black")
        if p < 0.05:
            ax.text(xi, 97, "*", ha="center", va="bottom", fontsize=14, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {out_path}")


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

    patients = load_patient_table(args.data_root)
    frames = load_reader_frames(args.data_root, patients)
    reader_df = build_reader_table(frames, args.data_root)
    cad_auc = cad_alone_curve(patients)[2] * 100.0

    print("Per-reader AUC (%):")
    print(reader_df[["reader", "unaided", "aided", "effect_size", "group"]]
          .round(1).to_string(index=False))
    print(f"CAD alone AUC: {cad_auc:.1f}%\n")

    print("Figure 4 (average reader ROC curves):")
    dbm_auc = load_dbm_auc(args.data_root)
    make_figure4(frames, patients, dbm_auc, args.out_dir / "figure4_roc_aided_unaided_ai.png")
    print("Figure 5 (effect size by sub-specialty):")
    make_figure5(reader_df, cad_auc, args.out_dir / "figure5_effect_size_by_specialty.png")
    print("Figure E1 (effect size by reader):")
    make_figureE1(reader_df, args.out_dir / "figureE1_effect_size_by_reader.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
