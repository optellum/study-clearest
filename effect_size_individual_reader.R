#!/usr/bin/env Rscript

# ==================================================================
# Per-reader DeLong AUC comparison: aided vs unaided
# ------------------------------------------------------------------
# For each reader, compares the ROC AUC of the patient likelihood-of-
# malignancy score with AI assistance (aided, LOM2) versus without
# (unaided, LOM1) using DeLong's test for two correlated ROC curves
# (pROC::roc.test, paired). Results are reported overall and split by
# study phase (S1, S2), together with the S2 - S1 change in the
# aided-unaided AUC difference (a learning / period effect). AUCs and
# their DeLong confidence intervals are expressed as percentages. All
# inference is analytic (DeLong); no bootstrap is used.
#
# Input format: CSV (clearest_results_patient_lom.csv) with one row per
# (reader, case, study phase): Case, Reader, LOM1 (unaided score), LOM2
# (aided score), L (case truth 0/1), aided_phase (S1 or S2).
#
# ==================================================================

suppressMessages({
  library(pROC)
})

mrmc_dir   <- "resources/clearest_main_study/mrmc"
input_csv  <- file.path(mrmc_dir, "all", "clearest_results_patient_lom.csv")
output_csv <- file.path(mrmc_dir, "plom_pvalue.csv")

# Readers who completed the study (participant ids; 12, 14, 16 are absent).
readers <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17)

CONF_LEVEL <- 0.95
round_n    <- 2

# DeLong comparison of aided vs unaided ROC AUC for one reader / phase.
delong_phase <- function(outcome, unaided, aided) {
  roc_un <- roc(outcome, unaided, quiet = TRUE)
  roc_ai <- roc(outcome, aided, quiet = TRUE)
  # One-sided test in the observed direction of the aided-unaided difference.
  alternative <- if (roc_ai$auc < roc_un$auc) "less" else "greater"
  test <- roc.test(roc_ai, roc_un, method = "delong", paired = TRUE,
                   alternative = alternative, conf.level = CONF_LEVEL)
  list(
    auc_un  = roc_un$auc * 100,
    auc_ai  = roc_ai$auc * 100,
    ci_un   = ci(roc_un) * 100,   # DeLong CI: c(lower, AUC, upper)
    ci_ai   = ci(roc_ai) * 100,
    diff    = (roc_ai$auc - roc_un$auc) * 100,
    diff_ci = test$conf.int * 100,                 # c(lower, upper)
    p       = test$p.value
  )
}

# Format an estimate and its CI bounds as "value (lo,hi)".
fmt <- function(est, lo, hi) {
  paste0(round(est, round_n), " (", round(lo, round_n), ",", round(hi, round_n), ")")
}

analyse_reader <- function(sub) {
  s1 <- sub[sub$aided_phase == "S1", ]
  s2 <- sub[sub$aided_phase == "S2", ]

  ov  <- delong_phase(sub$L, sub$LOM1, sub$LOM2)
  r1  <- delong_phase(s1$L, s1$LOM1, s1$LOM2)
  r2  <- delong_phase(s2$L, s2$LOM1, s2$LOM2)
  delta_s2_s1 <- round(round(r2$diff, round_n) - round(r1$diff, round_n), round_n)

  cat(sprintf("Reader %2d | aided %s  unaided %s  diff %s  p=%.4f\n",
              sub$Reader[1], fmt(ov$auc_ai, ov$ci_ai[1], ov$ci_ai[3]),
              fmt(ov$auc_un, ov$ci_un[1], ov$ci_un[3]),
              fmt(ov$diff, ov$diff_ci[1], ov$diff_ci[2]), ov$p))

  data.frame(
    Reader               = sub$Reader[1],
    AUC_diff             = fmt(ov$diff, ov$diff_ci[1], ov$diff_ci[2]),
    AUC_Aided            = fmt(ov$auc_ai, ov$ci_ai[1], ov$ci_ai[3]),
    AUC_Unaided          = fmt(ov$auc_un, ov$ci_un[1], ov$ci_un[3]),
    pvalue               = round(ov$p, 4),
    delta_AUC_S2_S1_diff = delta_s2_s1,
    AUC_diff_S1          = fmt(r1$diff, r1$diff_ci[1], r1$diff_ci[2]),
    AUC_Aided_S1         = fmt(r1$auc_ai, r1$ci_ai[1], r1$ci_ai[3]),
    AUC_Unaided_S1       = fmt(r1$auc_un, r1$ci_un[1], r1$ci_un[3]),
    pvalue_S1            = round(r1$p, 4),
    AUC_diff_S2          = fmt(r2$diff, r2$diff_ci[1], r2$diff_ci[2]),
    AUC_Aided_S2         = fmt(r2$auc_ai, r2$ci_ai[1], r2$ci_ai[3]),
    AUC_Unaided_S2       = fmt(r2$auc_un, r2$ci_un[1], r2$ci_un[3]),
    pvalue_S2            = round(r2$p, 4),
    stringsAsFactors     = FALSE
  )
}

df <- read.csv(input_csv, header = TRUE, sep = ",")
results <- do.call(rbind, lapply(readers, function(r) analyse_reader(df[df$Reader == r, ])))

write.csv(results, output_csv, row.names = FALSE)
cat(sprintf("\nWrote %d readers to %s\n", nrow(results), output_csv))
