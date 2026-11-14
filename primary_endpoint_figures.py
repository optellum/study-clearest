"""Primary endpoint figures for the CLEAREST main study manuscript (Radiology).

This script reproduces the primary-endpoint figures of the CLEAREST multi-reader
multi-case (MRMC) study. The endpoint is the change in reader performance,
measured by the area under the ROC curve (AUC) of the patient-level likelihood of
malignancy (``patient_lom``), when reading is aided by the Lung Cancer Prediction
(LCP) AI ("With CAD") versus unaided ("Without CAD"). The standalone AI score is
shown as a "CAD alone" reference.

Figures produced
----------------
Figure 3  -- Average reader ROC curves: Without CAD, With CAD, and CAD alone. The
             reader-arm AUC and 95% CI shown are the DBM (Wilcoxon FOM, RRRC) values
             from effect_size_cohort.R; CAD alone is an empirical AUC (bootstrap CI).
             Shaded areas are pointwise 95% bootstrap confidence bands. The reader
             curves resample readers and patients; CAD alone resamples patients.
Figure 4  -- Effect size by reader sub-specialty. Mean unaided AUC, mean aided AUC,
             and their difference (the AI effect size) for cardiothoracic
             specialists, other radiologists, and all readers combined.
Figure E1 -- Effect size for every reader (supplemental material). Per-reader
             unaided AUC, aided AUC, and their difference, with an asterisk marking
             readers whose aided-vs-unaided difference is statistically significant
             (p < 0.05, DeLong test).

All figures are computed over all patients and are based on the patient-level
likelihood of malignancy. Per-reader significance (Figure E1 asterisks) comes from
the paired DeLong test in ``effect_size_individual_reader.R`` (mrmc/plom_pvalue.csv).

Manuscript legends
------------------
Caption text copied from the clean resubmission manuscript and supplement (CLEAREST_manuscript_121126, CLEAREST_supplemental_121126).

Figure 3. Average reader receiver operating characteristic curves for patient-level
discrimination of lung cancer (240 examinations) under two reading conditions: without
computer-aided diagnosis (CAD) and with CAD. Average area under the receiver operating
characteristic curve (AUC) was computed across 15 readers participating in the study
using the Dorfman-Berbaum-Metz method, which accounts for the multi-reader multi-case
study design. The CAD only performance is also shown. Shaded bands show pointwise 95%
CIs for the unaided and aided curves (bootstrap resampling of cases and readers); the
device-alone AUC and its 95% CI are given in Table E7. CI = confidence interval.

Figure 4. Average reader area under the receiver operating characteristic curve (AUC) by
radiology sub-specialty (Cardiothoracic n = 10, Other/general n = 5, All readers n =
15), estimated via the Dorfman-Berbaum-Metz method. Bars show unaided AUC (light) and
the increment with CAD assistance (dark), stacked to aided AUC; the dashed line marks
CAD-alone AUC (0.93). Values above each bar are aided AUC, and values within the light
bars are unaided AUC. Effect sizes were significant for all three groups (P < .001). CAD
= computer-aided detection and diagnosis.

Figure E1. Per-reader change in average area under the receiver operating characteristic
curve (ΔAUC, aided minus unaided) for the Patient Likelihood of Malignancy rating task
across 15 readers, ordered by effect size. All readers improved with CAD assistance.
Estimated under the Dorfman-Berbaum-Metz framework on the full 240 examinations cohort.
* One-sided DeLong’s test, P < .05.

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
  mrmc/cohort_auc_by_treatment.csv   (Figure 3 reader-arm AUC/CI)
      Per-treatment DBM AUC and RRRC 95% CI, written by effect_size_cohort.R.
      Run that R script before this one.
  mrmc/plom_pvalue.csv               (Figure E1 asterisks)
      Per-reader paired DeLong P value (columns Reader = 0..17, pvalue), written by
      effect_size_individual_reader.R. Run that R script before this one.

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


def load_reader_pvalues(data_root: Path) -> dict[str, float]:
    """Per-reader DeLong P value (aided vs unaided AUC) from effect_size_individual_reader.R.
    Used only to annotate significant readers in Figure E1."""
    d = pd.read_csv(data_root / "mrmc" / "plom_pvalue.csv")
    p = {f"MP{int(r)}": float(v) for r, v in zip(d["Reader"], d["pvalue"])}
    missing = [r for r in READERS if r not in p]
    if missing:
        raise ValueError(f"plom_pvalue.csv has no P value for {missing}")
    return p


# Score columns: the reader's malignancy score and the standalone AI score.
READER_SCORE_COL = "patient_lom"          # patient-level likelihood of malignancy (0-100)
AI_SCORE_COL = "zclcp_lcp_score_max_y"     # standalone LCP AI score, "CAD alone"

N_BOOTSTRAPS = 2000  # bootstrap iterations for the AUC confidence intervals
BOOTSTRAP_SEED = 42  # fixed seed for reproducible bootstrap CIs

# Stacked bar colors (Figures 4 and E1).
COLOR_UNAIDED = "#d3d3d3"      # "Without CAD" base bar
COLOR_IMPROVEMENT = "#5a5a5a"  # aided-minus-unaided improvement, stacked on top

# ROC curve colors (Figure 3).
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
        data_root / "dataset" / "clearest_patient_data_240_main_study_max_lcp_score.csv")


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
    """AUC (0-1) of patient_lom for one reader's aided (1) or unaided (0) reads."""
    arm = frame[frame["is_aided_encord"] == is_aided]
    return roc_auc_score(arm["is_cancer"], arm[READER_SCORE_COL])


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
            "DBM per-treatment AUC/CI shown on Figure 3.")
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


def _interp_tpr(labels, scores):
    """Empirical ROC curve interpolated onto MEAN_FPR (TPR forced to 0 at FPR = 0)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    interp = np.interp(MEAN_FPR, fpr, tpr)
    interp[0] = 0.0
    return interp


def reader_roc_bands(frames, n_bootstraps=N_BOOTSTRAPS, ci=0.95, seed=BOOTSTRAP_SEED):
    """Pointwise bootstrap bands for the reader-averaged ROC curves (both arms).

    Two-way (random-reader, random-case) bootstrap, matching the RRRC inference of
    the DBM AUC CIs shown in the legend. Each replicate resamples the 15 readers and
    the patients with replacement. The same patients are used for every reader and
    for both arms, so the paired design is kept. The replicate's reader-averaged TPR
    is recomputed on MEAN_FPR. The band is the pointwise 2.5th-97.5th percentile of
    the replicates. Replicates where a resampled reader has only one truth class in
    an arm are skipped.

    Returns ``{0: (lo, hi), 1: (lo, hi)}`` keyed by is_aided_encord.
    """
    rng = np.random.default_rng(seed)
    # Per reader and arm: patient_lom and truth indexed by stage-1 study_id.
    arms = {
        (reader, is_aided): f[f["is_aided_encord"] == is_aided].set_index("study_id")[
            ["is_cancer", READER_SCORE_COL]]
        for reader, f in frames.items() for is_aided in (0, 1)
    }
    patient_ids = np.array(sorted({pid for a in arms.values() for pid in a.index}))
    readers = list(frames)
    curves = {0: [], 1: []}
    for _ in range(n_bootstraps):
        boot_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot_readers = rng.choice(readers, size=len(readers), replace=True)
        rep = {0: [], 1: []}
        ok = True
        for reader in boot_readers:
            for is_aided in (0, 1):
                sample = arms[(reader, is_aided)].reindex(boot_ids).dropna()
                if sample["is_cancer"].nunique() < 2:
                    ok = False
                    break
                rep[is_aided].append(_interp_tpr(sample["is_cancer"], sample[READER_SCORE_COL]))
            if not ok:
                break
        if not ok:
            continue
        for is_aided in (0, 1):
            mean_tpr = np.mean(rep[is_aided], axis=0)
            mean_tpr[-1] = 1.0
            curves[is_aided].append(mean_tpr)
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    return {arm: tuple(np.percentile(np.array(c), q, axis=0)) for arm, c in curves.items()}


def cad_alone_band(patients: pd.DataFrame, n_bootstraps=N_BOOTSTRAPS, ci=0.95, seed=BOOTSTRAP_SEED):
    """Pointwise case-bootstrap band for the CAD-alone ROC curve on MEAN_FPR.

    CAD alone is a single rater, so only patients are resampled (consistent with
    its case-bootstrap AUC CI). Returns ``(lo, hi)``.
    """
    rng = np.random.default_rng(seed)
    ai = patients.dropna(subset=[AI_SCORE_COL, "is_cancer"])
    labels = ai["is_cancer"].to_numpy()
    scores = ai[AI_SCORE_COL].to_numpy()
    curves = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        tpr = _interp_tpr(labels[idx], scores[idx])
        tpr[-1] = 1.0
        curves.append(tpr)
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    lo, hi = np.percentile(np.array(curves), q, axis=0)
    return lo, hi


def build_reader_table(frames, data_root: Path) -> pd.DataFrame:
    """Per-reader unaided/aided AUC, effect size, and specialty group."""
    demographics = pd.read_csv(data_root / "dataset" / "participant_demographic.csv")
    reader_p = load_reader_pvalues(data_root)
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
            "p_value": reader_p[reader],
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
def make_figure3(frames, patients: pd.DataFrame, dbm_auc: dict, out_path: Path) -> None:
    """Figure 3 -- average reader ROC curves (Without CAD, With CAD, CAD alone).

    The reader-arm ROC curves are the across-reader mean TPR, but their AUC and 95%
    CI are the DBM (Wilcoxon FOM) reader-averaged AUC with the random-reader
    random-case CI, taken from ``effect_size_cohort.R`` (via ``load_dbm_auc``) so
    they match the manuscript. CAD alone is the standalone AI: an empirical AUC with
    a case-bootstrap CI (a single rater is not a DBM treatment).

    Shaded areas are pointwise 95% bootstrap confidence bands on the FPR grid. The
    reader arms use a two-way bootstrap that resamples readers and patients (see
    ``reader_roc_bands``). CAD alone resamples patients only (``cad_alone_band``).
    A second version, ``<out_path stem>_reader_bands.png``, shades only the two
    reader arms (the CAD-alone curve is drawn without its band).
    """
    mean_tpr_unaided = mean_reader_roc(frames, is_aided=0)
    mean_tpr_aided = mean_reader_roc(frames, is_aided=1)
    mean_auc_u, lo_u, hi_u = dbm_auc["unaided"]
    mean_auc_a, lo_a, hi_a = dbm_auc["aided"]
    fpr_ai, tpr_ai, auc_ai, lo_ai, hi_ai = cad_alone_curve(patients)
    # AUCs are displayed on the 0-1 scale with two decimals (Radiology style, e.g. 0.88).
    print(f"  Without CAD AUC {mean_auc_u:.3f} ({lo_u:.3f}, {hi_u:.3f})")
    print(f"  With CAD    AUC {mean_auc_a:.3f} ({lo_a:.3f}, {hi_a:.3f})")
    print(f"  CAD alone   AUC {auc_ai:.3f} ({lo_ai:.3f}, {hi_ai:.3f})")

    # Pointwise 95% bootstrap bands: two-way (readers + patients) for the reader
    # arms, patients only for CAD alone. Drawn first so the curves sit on top.
    bands = reader_roc_bands(frames)
    lo_band_ai, hi_band_ai = cad_alone_band(patients)

    # Two versions: all three bands (out_path), and reader-arm bands only
    # (CAD-alone curve drawn without its band; "_reader_bands" suffix).
    reader_only_path = out_path.with_name(f"{out_path.stem}_reader_bands{out_path.suffix}")
    for path, show_cad_band in ((out_path, True), (reader_only_path, False)):
        fig, ax = plt.subplots(figsize=(6, 6))
        if show_cad_band:
            ax.fill_between(MEAN_FPR, lo_band_ai, hi_band_ai, color=COLOR_ROC_AI, alpha=0.12, lw=0)
        ax.fill_between(MEAN_FPR, *bands[0], color=COLOR_ROC_UNAIDED, alpha=0.15, lw=0)
        ax.fill_between(MEAN_FPR, *bands[1], color=COLOR_ROC_AIDED, alpha=0.15, lw=0)
        ax.plot(fpr_ai, tpr_ai, color=COLOR_ROC_AI, lw=2, linestyle="dashed",
                label=f"CAD alone: AUC = {auc_ai:.2f} (95% CI: {lo_ai:.2f}, {hi_ai:.2f})")
        ax.plot(MEAN_FPR, mean_tpr_unaided, color=COLOR_ROC_UNAIDED, lw=2,
                label=f"Without CAD: AUC = {mean_auc_u:.2f} (95% CI: {lo_u:.2f}, {hi_u:.2f})")
        ax.plot(MEAN_FPR, mean_tpr_aided, color=COLOR_ROC_AIDED, lw=2,
                label=f"With CAD: AUC = {mean_auc_a:.2f} (95% CI: {lo_a:.2f}, {hi_a:.2f})")
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
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")


def make_figure4(reader_df: pd.DataFrame, cad_auc: float, out_path: Path) -> None:
    """Figure 4 -- effect size by reader sub-specialty."""
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
        print(f"  {group:<15} unaided {mean_ci(sub['unaided'])[0]:.3f} "
              f"aided {mean_ci(sub['aided'])[0]:.3f} "
              f"effect {sub['aided'].mean() - sub['unaided'].mean():+.3f}")
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
               label=f"CAD alone ({cad_auc:.2f})")

    fontsize = 14
    ax.set_ylabel("Average area under the ROC curve", fontsize=fontsize)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.round(np.arange(0.5, 1.01, 0.1), 1))
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.legend(frameon=False, fontsize=fontsize, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels: unaided AUC below the grey bar, aided AUC above the stack.
    for xi, base, total in zip(x, unaided, aided):
        ax.text(xi, base - 0.01, f"{base:.2f}", ha="center", va="top",
                fontsize=fontsize, color="black")
        ax.text(xi, total + 0.01, f"{total:.2f}", ha="center", va="bottom",
                fontsize=fontsize, color="black")

    ax.set_xticks([])
    ax.xaxis.set_ticks_position("none")
    for xi, group in zip(x, groups):
        ax.text(xi, 0.49, labels[group], ha="center", va="top",
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
    ax.set_ylabel("Area under the ROC curve", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.round(np.arange(0.5, 1.01, 0.1), 1))
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
        ax.text(xi, base - 0.01, f"{base:.2f}", ha="center", va="top",
                fontsize=8, color="black")
        ax.text(xi, total + 0.01, f"{total:.2f}", ha="center", va="bottom",
                fontsize=8, color="black")
        if p < 0.05:
            ax.text(xi, 0.97, "*", ha="center", va="bottom", fontsize=14, color="black")

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
    cad_auc = cad_alone_curve(patients)[2]

    print("Per-reader AUC:")
    print(reader_df[["reader", "unaided", "aided", "effect_size", "group"]]
          .round(3).to_string(index=False))
    print(f"CAD alone AUC: {cad_auc:.3f}\n")

    print("Figure 3 (average reader ROC curves):")
    dbm_auc = load_dbm_auc(args.data_root)
    make_figure3(frames, patients, dbm_auc, args.out_dir / "figure3_roc_aided_unaided_ai.png")
    print("Figure 4 (effect size by sub-specialty):")
    make_figure4(reader_df, cad_auc, args.out_dir / "figure4_effect_size_by_specialty.png")
    print("Figure E1 (effect size by reader):")
    make_figureE1(reader_df, args.out_dir / "figureE1_effect_size_by_reader.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
