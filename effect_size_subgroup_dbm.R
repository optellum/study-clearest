#!/usr/bin/env Rscript

# ==================================================================
# DBM effect size by subgroup (Table 3 input)
# ------------------------------------------------------------------
# Runs the Dorfman-Berbaum-Metz (Hillis-modified) multi-reader
# multi-case (MRMC) analysis of the AI effect, aided vs unaided, on each
# subgroup dataset and writes one row per dataset to
# mrmc/dataset_effect_size_analysis.csv. primary_endpoint_tables.py
# formats Table 3 from this file; the P for interaction column comes
# from effect_size_subgroup_interaction.R.
#
# Method (per dataset): RJafroc StSignificanceTesting, figure of merit =
# Wilcoxon (empirical) ROC AUC of the patient likelihood of malignancy,
# method "DBM", random-reader random-case (RRRC) analysis, 95% CIs.
# Treatments are trt0 = unaided and trt1 = aided; the effect size is the
# aided - unaided AUC difference. Within each subgroup the P value is the
# RRRC F test of the treatment effect (unadjusted).
#
# Subgroups:
#   - reader subgroups (all 240 cases, a subset of readers): sub-specialty,
#     experience (years of practice; also years since certification),
#     prior AI use, and the four post-study attitude scales;
#   - case subgroups (all 15 readers, a subset of cases): index-nodule
#     long-axis diameter (pre-specified <9 / 9 to <15 / >=15 mm, unrounded),
#     attenuation, margin, and screening vs diagnostic CT.
#   The remaining datasets (full cohort, thresholds, difficulty) are kept
#   so the file holds every DBM result in one place; Table 3 does not use
#   them.
#
# Input format: one iMRMC text file per dataset under
# resources/clearest_main_study/mrmc/all/<dataset>_patient_lom.imrmc
# (comma-delimited; read by RJafroc DfReadDataFile): two modalities
# trt0 = unaided, trt1 = aided; one score (patient likelihood of
# malignancy, 0-100) per reader x case; binary case truth (0 = benign or
# no nodule, 1 = malignant). The subgroup files are produced upstream by
# create_imrmc_files() in MAIN_create_immr_files.ipynb.
#
# Output: resources/clearest_main_study/mrmc/dataset_effect_size_analysis.csv
#   effect_size, aided_trt, unaided_trt: percentages, "estimate (lo,hi)";
#   *_raw columns: the same values unrounded on the 0-1 AUC scale;
#   pvalue_rrrc (rounded) and pvalue_rrrc_raw; var_tr, var_tc, var_err:
#   DBM variance components.
#
# Run from the repository root that contains resources/clearest_main_study:
#   Rscript effect_size_subgroup_dbm.R
#
# Note: findings of this reader study, not a device performance claim.
# ==================================================================

suppressMessages({
  library(RJafroc)
})

mrmc_dir <- "resources/clearest_main_study/mrmc"
folder   <- "all"
endpoint <- "_patient_lom"
out_path <- file.path(mrmc_dir, "dataset_effect_size_analysis.csv")

FOM            <- "Wilcoxon"
method         <- "DBM"
alpha          <- 0.05    # 95% CIs
analysisOption <- "RRRC"  # random readers, random cases
round_n        <- 2

datasets <- c(
  # not used by Table 3
  "clearest_results", "clearest_results_5", "clearest_results_10",
  "clearest_results_nodule", "clearest_results_FG_easy", "clearest_results_FG_difficult",
  # reader subgroups
  "clearest_results_specialty_yes", "clearest_results_specialty_no",
  "clearest_results_experience_yes", "clearest_results_experience_no",
  "clearest_results_certification_yes", "clearest_results_certification_no",
  "clearest_results_use_ai_yes", "clearest_results_use_ai_no",
  "clearest_results_confirmation_bias_yes", "clearest_results_confirmation_bias_no",
  "clearest_results_automation_bias_yes", "clearest_results_automation_bias_no",
  "clearest_results_aversion_bias_yes", "clearest_results_aversion_bias_no",
  "clearest_results_trust_in_lcp_yes", "clearest_results_trust_in_lcp_no",
  # case subgroups: pre-specified index-nodule size groups
  "clearest_results_nodule_small", "clearest_results_nodule_medium",
  "clearest_results_nodule_large",
  # case subgroups: attenuation, margin, CT type
  "clearest_results_nodule_solid", "clearest_results_nodule_part_solid",
  "clearest_results_nodule_ggo",
  "clearest_results_nodule_spiculated", "clearest_results_nodule_non_spiculated",
  "clearest_results_screening", "clearest_results_not_screening"
)

# "estimate (lo,hi)" as percentages, 2 decimals.
fmt <- function(est, lo, hi) {
  paste0(round(est * 100, round_n), " (", round(lo * 100, round_n), ",",
         round(hi * 100, round_n), ")")
}

dbm_row <- function(dataset) {
  file <- file.path(mrmc_dir, folder, paste0(dataset, endpoint, ".imrmc"))
  rocData <- DfReadDataFile(file, format = "iMRMC", delimiter = ",")
  st <- StSignificanceTesting(rocData, FOM = FOM, FPFValue = 0.2, alpha = alpha,
                              method = method, analysisOption = analysisOption)
  fom  <- st$FOMs
  rrrc <- st$RRRC
  # trtMeanDiffs and ciDiffTrt are trt0 - trt1; flip the sign for aided - unaided.
  es    <- -fom$trtMeanDiffs["trt0-trt1", ]
  es_lo <- -rrrc$ciDiffTrt["trt0-trt1", "CIUpper"]
  es_hi <- -rrrc$ciDiffTrt["trt0-trt1", "CILower"]
  ci    <- rrrc$ciAvgRdrEachTrt
  p     <- rrrc$FTests["Treatment", "p"]
  data.frame(
    experiment_case = folder, dataset = dataset, endpoint = endpoint, aided_phase = "",
    effect_size = fmt(es, es_lo, es_hi),
    aided_trt   = fmt(fom$trtMeans["trt1", ], ci["trt1", "CILower"], ci["trt1", "CIUpper"]),
    unaided_trt = fmt(fom$trtMeans["trt0", ], ci["trt0", "CILower"], ci["trt0", "CIUpper"]),
    pvalue_rrrc = round(p, round_n + 1),
    var_tr  = st$ANOVA$VarCom["VarTR", ],
    var_tc  = st$ANOVA$VarCom["VarTC", ],
    var_err = st$ANOVA$VarCom["VarErr", ],
    es_raw = es, es_lo_raw = es_lo, es_hi_raw = es_hi,
    aided_raw = fom$trtMeans["trt1", ],
    aided_lo_raw = ci["trt1", "CILower"], aided_hi_raw = ci["trt1", "CIUpper"],
    unaided_raw = fom$trtMeans["trt0", ],
    unaided_lo_raw = ci["trt0", "CILower"], unaided_hi_raw = ci["trt0", "CIUpper"],
    pvalue_rrrc_raw = p
  )
}

rows <- lapply(datasets, function(d) {
  r <- dbm_row(d)
  cat(sprintf("%-42s effect %-20s aided %-22s unaided %-22s P = %.3g\n",
              d, r$effect_size, r$aided_trt, r$unaided_trt, r$pvalue_rrrc_raw))
  r
})
out <- do.call(rbind, rows)
rownames(out) <- NULL
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("Wrote %s (%d datasets)\n", out_path, nrow(out)))
