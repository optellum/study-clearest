#!/usr/bin/env Rscript

# ==================================================================
# DBM effect-size analysis: aided vs unaided (primary endpoint)
# ------------------------------------------------------------------
# Standard Dorfman-Berbaum-Metz (Hillis-modified) multi-reader
# multi-case (MRMC) significance test of the AI effect on reader
# performance for the full CLEAREST cohort (dataset "clearest_results").
#
# The figure of merit is the Wilcoxon (empirical) ROC AUC of the
# patient likelihood-of-malignancy score. Treatments are trt0 = unaided
# and trt1 = aided; the reported AI effect size is the aided - unaided
# AUC difference with its random-reader random-case (RRRC) 95% CI.
# Effect size and per-treatment AUCs are expressed as percentages.
#
# The iMRMC input is produced upstream by create_imrmc_files() in
# MAIN_create_immr_files.ipynb.
#
# Input format: iMRMC text file (comma-delimited; read by RJafroc
# DfReadDataFile). Two modalities trt0 = unaided, trt1 = aided; one score
# (patient likelihood of malignancy) per reader x case; binary case truth
# (0 = benign, 1 = malignant). 15 readers, 240 cases.
# ==================================================================

suppressMessages({
  library(RJafroc)
})

mrmc_dir <- "resources/clearest_main_study/mrmc"
folder   <- "all"
endpoint <- "_patient_lom"
dataset  <- "clearest_results"

FOM            <- "Wilcoxon"
method         <- "DBM"
alpha          <- 0.05    # 95% CI on the aided - unaided effect size
analysisOption <- "RRRC"  # random readers, random cases
round_n        <- 2

# Format an estimate and its CI as "value (lo,hi)", scaled to a percentage.
fmt <- function(est, lo, hi) {
  paste0(round(est * 100, round_n), " (",
         round(lo * 100, round_n), ",", round(hi * 100, round_n), ")")
}

compute_effect_size <- function(dataset, endpoint) {

  file <- file.path(mrmc_dir, folder, paste0(dataset, endpoint, ".imrmc"))
  rocData <- DfReadDataFile(file, format = "iMRMC", delimiter = ",")

  st <- StSignificanceTesting(rocData, FOM = FOM, FPFValue = 0.2,
                              alpha = alpha, method = method,
                              analysisOption = analysisOption)
  fom  <- st$FOMs
  rrrc <- st$RRRC

  # Effect size is aided - unaided; trtMeanDiffs stores trt0 - trt1, so the
  # difference and its CI bounds are sign-flipped to express aided - unaided.
  effect_size <- fmt(-fom$trtMeanDiffs["trt0-trt1", ],
                     -rrrc$ciDiffTrt["trt0-trt1", "CIUpper"],
                     -rrrc$ciDiffTrt["trt0-trt1", "CILower"])
  aided_auc   <- fmt(fom$trtMeans["trt1", ],
                     rrrc$ciAvgRdrEachTrt["trt1", "CILower"],
                     rrrc$ciAvgRdrEachTrt["trt1", "CIUpper"])
  unaided_auc <- fmt(fom$trtMeans["trt0", ],
                     rrrc$ciAvgRdrEachTrt["trt0", "CILower"],
                     rrrc$ciAvgRdrEachTrt["trt0", "CIUpper"])

  list(
    dataset     = dataset,
    endpoint    = sub("^_", "", endpoint),
    effect_size = effect_size,
    aided_auc   = aided_auc,
    unaided_auc = unaided_auc,
    p_value     = rrrc$FTests["Treatment", "p"],
    var_tr      = st$ANOVA$VarCom["VarTR", ],
    var_tc      = st$ANOVA$VarCom["VarTC", ],
    var_err     = st$ANOVA$VarCom["VarErr", ],
    # Raw per-treatment AUC + RRRC CI (fractions), exported for Figure 4.
    unaided_val = fom$trtMeans["trt0", ],
    unaided_lo  = rrrc$ciAvgRdrEachTrt["trt0", "CILower"],
    unaided_hi  = rrrc$ciAvgRdrEachTrt["trt0", "CIUpper"],
    aided_val   = fom$trtMeans["trt1", ],
    aided_lo    = rrrc$ciAvgRdrEachTrt["trt1", "CILower"],
    aided_hi    = rrrc$ciAvgRdrEachTrt["trt1", "CIUpper"]
  )
}

res <- compute_effect_size(dataset, endpoint)

cat(sprintf("DBM effect-size analysis (aided vs unaided) -- dataset %s, endpoint %s\n",
            res$dataset, res$endpoint))
cat(sprintf("  Effect size (aided - unaided AUC, %%): %s\n", res$effect_size))
cat(sprintf("  Aided AUC (%%)  : %s\n", res$aided_auc))
cat(sprintf("  Unaided AUC (%%): %s\n", res$unaided_auc))
cat(sprintf("  RRRC treatment p-value: %.3g\n", res$p_value))
cat(sprintf("  Variance components: VarTR=%.3g  VarTC=%.3g  VarErr=%.3g\n",
            res$var_tr, res$var_tc, res$var_err))

# Export the per-treatment DBM AUC and RRRC CI (fractions) for the Figure 4
# legend in primary_endpoint_figures.py. trt0 = unaided, trt1 = aided.
auc_by_treatment <- data.frame(
  treatment = c("trt0", "trt1"),
  arm       = c("unaided", "aided"),
  auc       = c(res$unaided_val, res$aided_val),
  ci_lower  = c(res$unaided_lo, res$aided_lo),
  ci_upper  = c(res$unaided_hi, res$aided_hi)
)
auc_out <- file.path(mrmc_dir, "cohort_auc_by_treatment.csv")
write.csv(auc_by_treatment, auc_out, row.names = FALSE)
cat(sprintf("  Wrote per-treatment AUC/CI to %s\n", auc_out))
