#!/usr/bin/env Rscript

# ==================================================================
# Device alone versus readers (reviewer R1-3)
# ------------------------------------------------------------------
# Compares the standalone LCP AI ("device alone", "CAD alone") with the
# 15-reader panel, separately for unaided and aided reads. The figure of
# merit is the empirical (Wilcoxon) ROC AUC of the patient-level score
# over all 240 cases (95 malignant, 145 non-malignant).
#
# Method: RJafroc StSignificanceTestingCadVsRad, method "1T-RRRC"
# (Chakraborty; single-treatment Obuchowski-Rockette analysis with the
# device as a fixed reader). The device is reader 1 and the radiologists are
# readers 2..16 of a single-treatment dataset. The test is on the
# reader-averaged AUC minus the device AUC; readers and cases are random,
# the device is fixed. Covariances are jackknife estimates. One analysis
# uses the unaided reads (LOM1), one the aided reads (LOM2).
#
# Table E7 layout (as in the manuscript):
#   - Device alone: AUC with a case-bootstrap 95% CI (2,000 resamples of
#     patients, percentile), the method of the Figure 3 CAD-alone CI;
#   - Readers without / with CAD: reader-average AUC and 95% CI of the
#     primary DBM analysis (read from mrmc/cohort_auc_by_treatment.csv,
#     written by effect_size_cohort.R; run that script first), and the
#     1T-RRRC difference vs the device (readers minus device), 95% CI and P.
# The 1T-RRRC T statistic and df, the number of readers above the device and
# the reader AUC range (Table E7 note) are printed to the console.
#
# Patient-level device score: the highest per-nodule LCP score in the
# examination (zclcp_lcp_score_max_y, 0-1). None of the 240 cases has a
# missing device score. The Wilcoxon AUC depends only on the score ranks
# within a reader, so the 0-1 device scale and the 0-100 reader scale can be
# analysed together.
#
# Input format:
#   - CSV (clearest_results_patient_lom.csv): one row per (reader, case):
#     Case, Reader, LOM1 (unaided pLoM), LOM2 (aided pLoM), L (truth 0/1).
#   - CSV (clearest_patient_data_240_main_study_max_lcp_score.csv):
#     study_id_stage1 (= Case), is_cancer, zclcp_lcp_score_max_y.
#   - CSV (mrmc/cohort_auc_by_treatment.csv): treatment (trt0 / trt1), auc,
#     ci_lower, ci_upper, from effect_size_cohort.R.
#
# Output: resources/clearest_main_study/tables_resubmission/device_vs_readers.csv
#
# Manuscript caption (Table E7; copied from the clean resubmission supplement,
# CLEAREST_supplemental_121126):
#   Table E7. Comparison of readers with the device alone.
#   Note - Single-treatment random-reader random-case analysis
#   (RJafroc StSignificanceTestingCadVsRad, method 1T-RRRC): the
#   device was treated as a fixed reader, and readers and examinations
#   as random factors, using the Wilcoxon AUC over all 240
#   examinations (Appendix E8). Differences are readers minus device.
#   Reader-average AUCs (95% CI) are those of the primary
#   Dorfman-Berbaum-Metz analysis. The device-alone 95% CI is from
#   bootstrap resampling of patients (2,000 resamples), as for Figure
#   3 (main manuscript). Device score: highest per-nodule LCP score
#   per examination; every examination had a device score. Readers
#   with an AUC above the device: unaided, 0 of 15 (range, 0.83–0.92);
#   aided, 5 of 15 (range, 0.90-0.95).
#
# Note: analysis requested by a reviewer (R1-3). This is a finding of this
# reader study, not a device performance claim.
# ==================================================================

suppressMessages({
  library(RJafroc)
})

lom_path <- "resources/clearest_main_study/mrmc/all/clearest_results_patient_lom.csv"
pat_path <- "resources/clearest_main_study/dataset/clearest_patient_data_240_main_study_max_lcp_score.csv"
dbm_path <- "resources/clearest_main_study/mrmc/cohort_auc_by_treatment.csv"
out_path <- "resources/clearest_main_study/tables_resubmission/device_vs_readers.csv"

lom <- read.csv(lom_path)
pat <- read.csv(pat_path)[, c("study_id_stage1", "is_cancer", "zclcp_lcp_score_max_y")]

# One row per case: truth and device score. Case order: non-malignant, then malignant.
cases <- merge(unique(lom[, c("Case", "L")]), pat, by.x = "Case", by.y = "study_id_stage1")
stopifnot(nrow(cases) == 240, all(cases$L == cases$is_cancer),
          !anyNA(cases$zclcp_lcp_score_max_y))
cases <- cases[order(cases$L, cases$Case), ]
nl_cases <- cases$Case[cases$L == 0]
ll_cases <- cases$Case[cases$L == 1]
readers  <- sort(unique(lom$Reader))
stopifnot(length(readers) == 15)

# Ratings matrix [device + 15 readers, cases] for one arm.
ratings <- function(case_ids, col) {
  dev <- cases$zclcp_lcp_score_max_y[match(case_ids, cases$Case)]
  rdr <- t(sapply(readers, function(r) {
    d <- lom[lom$Reader == r, ]
    d[[col]][match(case_ids, d$Case)]
  }))
  m <- rbind(dev, rdr)
  stopifnot(!anyNA(m))
  m
}

# Df2RJafrocDataset leaves truthTableStr empty, but the CAD-vs-radiologist
# pseudovalue code needs it. Fill it as in RJafroc's dataset09: [1, reader, case, 1] = 1
# for non-diseased cases and [1, reader, case, 2] = 1 for diseased cases.
with_truth_table <- function(ds) {
  J  <- dim(ds$ratings$NL)[2]
  K  <- dim(ds$ratings$NL)[3]
  K2 <- dim(ds$ratings$LL)[3]
  K1 <- K - K2
  tt <- array(NA, dim = c(1, J, K, 2))
  tt[1, , 1:K1, 1] <- 1
  tt[1, , (K1 + 1):K, 2] <- 1
  ds$descriptions$truthTableStr <- tt
  ds
}

# Radiology P format: no leading zero; <.001; three decimals below .01.
fmt_p <- function(p) {
  if (p < 0.001) return("<.001")
  sub("^0", "", if (p < 0.01) sprintf("%.3f", p) else sprintf("%.2f", p))
}
fmt_ci <- function(est, lo, hi) sprintf("%.3f (%.3f, %.3f)", est, lo, hi)

# Manuscript format: two decimals, Unicode minus sign.
f2 <- function(x) gsub("-", "−", sprintf("%.2f", x), fixed = TRUE)
fmt_ci2 <- function(est, lo, hi) sprintf("%s (%s, %s)", f2(est), f2(lo), f2(hi))

# Device-alone AUC with a case-bootstrap 95% CI (2,000 resamples of patients; percentile),
# the method of the Figure 3 CAD-alone CI. R's random-number stream differs from the Python
# one in primary_endpoint_figures.py, so the third decimal can differ; check that both
# agree at the two decimals reported.
boot_auc_ci <- function(L, s, n_boot = 2000, seed = 42) {
  set.seed(seed)
  b <- replicate(n_boot, {
    i <- sample.int(length(L), replace = TRUE)
    if (length(unique(L[i])) < 2) NA else auc(L[i], s[i])
  })
  quantile(b, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
}
auc <- function(L, s) {
  pos <- s[L == 1]; neg <- s[L == 0]
  mean(outer(pos, neg, function(a, b) (a > b) + 0.5 * (a == b)))
}
dev_auc <- auc(cases$L, cases$zclcp_lcp_score_max_y)
ci_boot <- boot_auc_ci(cases$L, cases$zclcp_lcp_score_max_y)
cat(sprintf("Device-alone AUC: %s (case bootstrap 95%% CI)\n", fmt_ci(dev_auc, ci_boot[1], ci_boot[2])))

# Reader-average AUC and 95% CI from the primary DBM analysis (effect_size_cohort.R).
dbm <- read.csv(dbm_path)
rows <- list(data.frame(
  Condition = "Device alone",
  "Average AUC (95% CI)" = fmt_ci2(dev_auc, ci_boot[1], ci_boot[2]),
  "Difference vs device alone (95% CI)" = "—", "P value" = "—", check.names = FALSE))
for (arm in list(list(label = "Readers without CAD", col = "LOM1", trt = "trt0"),
                 list(label = "Readers with CAD",    col = "LOM2", trt = "trt1"))) {
  ds <- with_truth_table(Df2RJafrocDataset(ratings(nl_cases, arm$col),
                                            ratings(ll_cases, arm$col)))
  st <- StSignificanceTestingCadVsRad(ds, FOM = "Wilcoxon", method = "1T-RRRC")
  fom_rad <- st$fomRAD
  diff    <- st$avgDiffFom            # reader average - device
  ci_diff <- as.numeric(st$CIAvgDiffFom)
  d <- dbm[dbm$treatment == arm$trt, ]
  stopifnot(nrow(d) == 1, abs(d$auc - mean(fom_rad)) < 1e-10,
            abs(st$fomCAD - dev_auc) < 1e-10)
  cat(sprintf(paste0("%s: reader-average AUC %.3f (DBM CI %.3f, %.3f); minus device %s; ",
                     "T = %.2f, df = %.1f, P = %s; readers above the device: %d of 15 ",
                     "(range, %.2f-%.2f)\n"),
              arm$label, d$auc, d$ci_lower, d$ci_upper, fmt_ci(diff, ci_diff[1], ci_diff[2]),
              unname(st$Tstat), unname(st$df), fmt_p(st$pval), sum(fom_rad > st$fomCAD),
              min(fom_rad), max(fom_rad)))
  rows[[length(rows) + 1]] <- data.frame(
    Condition = arm$label,
    "Average AUC (95% CI)" = fmt_ci2(d$auc, d$ci_lower, d$ci_upper),
    "Difference vs device alone (95% CI)" = fmt_ci2(diff, ci_diff[1], ci_diff[2]),
    "P value" = fmt_p(st$pval), check.names = FALSE)
}
out <- do.call(rbind, rows)
write.csv(out, out_path, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("Wrote %s\n", out_path))
