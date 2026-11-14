#!/usr/bin/env Rscript

# ==================================================================
# Reader-group contrasts (reviewer R4-1, Table E6)
# ------------------------------------------------------------------
# The reader attributes overlap: all general/other sub-specialty readers
# are early-career and all senior readers are cardiothoracic. This script
# splits the 15 readers into the three groups that exist in the panel
#   - cardiothoracic, senior (>= 5 years)
#   - cardiothoracic, early-career (< 5 years)
#   - general/other, early-career (< 5 years)
# and reports, per group, the DBM reader-averaged AUC without and with CAD
# and the effect size (aided - unaided Wilcoxon AUC) with its 95% CI.
# It then compares the effect size between groups:
#   - every pair of groups (difference in mean effect size, 95% CI, P;
#     P also Holm-adjusted across the pairwise comparisons);
#   - one omnibus test across all groups.
# Early-career cardiothoracic vs general/other separates specialty from
# experience; senior vs early-career cardiothoracic separates experience from
# specialty.
#
# Method:
#   - Group AUCs and effect sizes: RJafroc StSignificanceTesting, DBM,
#     RRRC (Hillis-modified; jackknife), on the subset of readers in the
#     group and all 240 cases; same method as the subgroup rows of Table 3.
#   - Comparisons: the Table 3 interaction method
#     (effect_size_subgroup_interaction.R). DBM jackknife pseudovalues
#     from all 15 readers; per-reader mean of the aided - unaided
#     pseudovalue difference (= that reader's AUC difference); one-way
#     ANOVA on these reader effects. Pairwise: ANOVA between the two
#     groups compared (ddf = readers in the two groups - 2), with the
#     pooled-variance t 95% CI of the difference in mean effect size, which
#     matches that ANOVA. Omnibus: one-way ANOVA across all groups
#     (ddf = 15 - number of groups). With 1 to 5 readers per group the
#     comparisons are exploratory; a group of one reader contributes no
#     variance and its comparisons rest on the other group's variance.
#
# Experience = years of practice (experience subgroup files, exp_y >= 5;
# 5 senior vs 10 early-career readers), as in Table 3. All senior readers are
# cardiothoracic, so the panel splits into three groups of 5 readers.
#
# Input format (under resources/clearest_main_study/mrmc/all/):
#   - clearest_results_patient_lom.csv: one row per (reader, case):
#     Case, Reader (0..17), LOM1 (unaided pLoM), LOM2 (aided pLoM), L (truth 0/1).
#   - clearest_results_subgroup_<factor>_patient_lom.csv for factor in
#     specialty (cardiothoracic/other) and experience (ge5y/lt5y): the same rows plus a "subgroup" label. These are the reader groups used in Table 3.
#
# Output (under resources/clearest_main_study/tables_resubmission/):
#   - tableE6_reader_groups.csv: the Table E6 rows (reader group, readers,
#     average AUC without / with CAD, effect size, P value);
#   - tableE6_comparisons.csv: the comparisons reported in the Table E6 note
#     (difference in effect size, 95% CI, P value, Holm-adjusted P value) and
#     the omnibus F test.
#
# Manuscript caption (Table E6; copied from the clean resubmission supplement,
# CLEAREST_supplemental_121126):
#   Table E6. PLoM effect size by reader group defined by
#   sub-specialty and experience.
#   Note - Experience is years of independent CT interpretation
#   (early-career, <5 years; senior, ≥5 years). All general/other
#   sub-specialty readers were early-career and all senior readers
#   were cardiothoracic (Table E2), so there is no senior
#   general/other group. Readers without a confirmation-bias tendency
#   comprised four general/other readers and one senior cardiothoracic
#   reader. Average AUC and effect size (AUC difference, with minus
#   without CAD) are estimated by the DBM method on all 240
#   examinations within each group; the P value tests the effect size
#   within each group. Comparisons of effect sizes (first minus second
#   group; 95% CI; P value, with Holm-adjusted P value in
#   parentheses):
#   senior vs early-career cardiothoracic: −0.005 (−0.030, 0.020), P =
#   .67 (.67);
#   senior cardiothoracic vs early-career general/other: −0.031
#   (−0.052, −0.011), P = .008 (.02);
#   early-career cardiothoracic vs early-career general/other: −0.026
#   (−0.048, −0.005), P = .02 (.045).
#   Across the three groups, F(2, 12) = 6.01, P = .02. Comparisons
#   used one-way ANOVA on per-reader DBM pseudovalue effect sizes (as
#   for the Table 3 interaction P values; 8 denominator degrees of
#   freedom), with pooled-variance 95% CIs; Holm adjustment is across
#   the three pairwise comparisons. All comparisons are exploratory.
#   AUC = area under the receiver operating characteristic curve, CAD
#   = computer-aided detection and diagnosis, DBM =
#   Dorfman-Berbaum-Metz, PLoM = patient likelihood of malignancy.
#
# Note: analysis requested by a reviewer (R4-1); exploratory,
# descriptive findings of this reader study, not a device performance
# claim.
# ==================================================================

suppressMessages({
  library(RJafroc)
})

mrmc_dir <- "resources/clearest_main_study/mrmc/all"
out_path  <- "resources/clearest_main_study/tables_resubmission/tableE6_reader_groups.csv"
comp_path <- "resources/clearest_main_study/tables_resubmission/tableE6_comparisons.csv"

lom <- read.csv(file.path(mrmc_dir, "clearest_results_patient_lom.csv"))

# Reader -> label for one subgroup factor.
reader_labels <- function(factor_name) {
  d <- read.csv(file.path(mrmc_dir,
                          paste0("clearest_results_subgroup_", factor_name, "_patient_lom.csv")))
  u <- unique(d[, c("Reader", "subgroup")])
  stopifnot(!anyDuplicated(u$Reader))
  setNames(u$subgroup, u$Reader)
}
spec <- reader_labels("specialty")

readers <- sort(unique(lom$Reader))
stopifnot(length(readers) == 15)

# Full dataset from the iMRMC file (trt0 = unaided, trt1 = aided). Its reader
# IDs are positions 0..14 in sorted MP order; check that by matching each
# position's unaided and aided Wilcoxon AUC with the AUC from the csv.
full <- DfReadDataFile(file.path(mrmc_dir, "clearest_results_patient_lom.imrmc"),
                       format = "iMRMC", delimiter = ",")
auc <- function(s, L) {
  pos <- s[L == 1]; neg <- s[L == 0]
  mean(outer(pos, neg, function(a, b) (a > b) + 0.5 * (a == b)))
}
fom_full <- UtilFigureOfMerit(full, FOM = "Wilcoxon")       # [trt, reader]
for (j in seq_along(readers)) {
  d <- lom[lom$Reader == readers[j], ]
  stopifnot(abs(fom_full[1, j] - auc(d$LOM1, d$L)) < 1e-10,
            abs(fom_full[2, j] - auc(d$LOM2, d$L)) < 1e-10)
}
dataset_for <- function(rdrs) {
  if (length(rdrs) == length(readers)) return(full)
  DfExtractDataset(full, rdrs = match(rdrs, readers))
}

fmt_ci <- function(est, lo, hi) sprintf("%.2f (%.2f, %.2f)", est, lo, min(hi, 1))
fmt_ci3 <- function(est, lo, hi) sprintf("%.3f (%.3f, %.3f)", est, lo, hi)
fmt_p <- function(p) {
  if (p < 0.001) return("<.001")
  if (p > 0.99) return(">.99")
  # Decide on decimals after rounding, so .0099x prints as .01, not .010. Three decimals
  # also when two would round a P below .05 up to .05 (e.g. .045).
  three <- round(p, 3) < 0.01 || (p < 0.05 && round(p, 2) >= 0.05)
  sub("^0", "", if (three) sprintf("%.3f", p) else sprintf("%.2f", p))
}

# DBM RRRC for a set of readers.
dbm <- function(rdrs) {
  st <- StSignificanceTesting(dataset_for(rdrs), FOM = "Wilcoxon", method = "DBM",
                              analysisOption = "RRRC")
  ci_trt <- st$RRRC$ciAvgRdrEachTrt
  ci_dif <- st$RRRC$ciDiffTrt          # trt0 - trt1
  list(
    unaided = fmt_ci(ci_trt["trt0", "Estimate"], ci_trt["trt0", "CILower"], ci_trt["trt0", "CIUpper"]),
    aided   = fmt_ci(ci_trt["trt1", "Estimate"], ci_trt["trt1", "CILower"], ci_trt["trt1", "CIUpper"]),
    effect  = fmt_ci(-ci_dif[1, "Estimate"], -ci_dif[1, "CIUpper"], -ci_dif[1, "CILower"]),
    effect_raw = -ci_dif[1, "Estimate"],
    p = fmt_p(ci_dif[1, "PrGTt"]))
}

# Per-reader AI effect from the DBM pseudovalues of all 15 readers
# (row mean of aided - unaided pseudovalues = the reader's AUC difference).
pv <- UtilPseudoValues(full, FOM = "Wilcoxon")$jkPseudoValues
theta <- setNames(rowMeans(pv[2, , ] - pv[1, , ]), readers)

# Omnibus one-way ANOVA of reader effects across groups (named list of reader vectors).
omnibus <- function(groups) {
  unit <- data.frame(theta = unlist(lapply(groups, function(g) theta[as.character(g)])),
                     grp = factor(rep(names(groups), sapply(groups, length))))
  a <- summary(aov(theta ~ grp, data = unit))[[1]]
  list(F = a["grp", "F value"], df1 = a["grp", "Df"], df2 = a["Residuals", "Df"],
       p = a["grp", "Pr(>F)"])
}

# Two-group contrast on reader effects: ANOVA P and pooled-t 95% CI of the difference.
contrast <- function(g1, g2) {
  x1 <- theta[as.character(g1)]; x2 <- theta[as.character(g2)]
  unit <- data.frame(theta = c(x1, x2),
                     grp = factor(rep(c("a", "b"), c(length(x1), length(x2)))))
  p <- summary(aov(theta ~ grp, data = unit))[[1]]["grp", "Pr(>F)"]
  tt <- t.test(x1, x2, var.equal = TRUE)
  stopifnot(abs(tt$p.value - p) < 1e-10)
  list(diff = fmt_ci3(mean(x1) - mean(x2), tt$conf.int[1], tt$conf.int[2]), p = p,
       df = length(x1) + length(x2) - 2)
}

# Manuscript number format: Unicode minus sign for negative values.
uminus <- function(s) gsub("-", "−", s, fixed = TRUE)

expl <- reader_labels("experience")
key <- as.character(readers)
is_ct <- spec[key] == "cardiothoracic"
senior <- expl[key] == "ge5y"
# Group label, plus the short specialty / experience words used to name comparisons.
groups <- list(
  list(label = "Cardiothoracic, senior (≥5 y)",       spec = "cardiothoracic", exp = "senior",
       readers = readers[is_ct & senior]),
  list(label = "Cardiothoracic, early-career (<5 y)",     spec = "cardiothoracic", exp = "early-career",
       readers = readers[is_ct & !senior]),
  list(label = "General/other, senior (≥5 y)",        spec = "general/other",  exp = "senior",
       readers = readers[!is_ct & senior]),
  list(label = "General/other, early-career (<5 y)",      spec = "general/other",  exp = "early-career",
       readers = readers[!is_ct & !senior]))
groups <- Filter(function(g) length(g$readers) > 0, groups)
groups[[length(groups) + 1]] <- list(label = "All readers", readers = readers)

table_rows <- list()
cat("\nExperience = years of practice\n")
for (g in groups) {
  r <- g$readers
  res <- if (length(r) >= 2) dbm(r) else {
    # A single reader: that reader's own AUCs, no CI.
    j <- match(r, readers)
    list(unaided = sprintf("%.2f", fom_full[1, j]), aided = sprintf("%.2f", fom_full[2, j]),
         effect = sprintf("%.2f", fom_full[2, j] - fom_full[1, j]), p = "",
         effect_raw = fom_full[2, j] - fom_full[1, j])
  }
  cat(sprintf("  %-38s n=%2d  %s  unaided %s  aided %s  effect %s [%.4f]\n",
              g$label, length(r), paste0("MP", r, collapse = " "), res$unaided, res$aided,
              res$effect, res$effect_raw))
  table_rows[[length(table_rows) + 1]] <- data.frame(
    "Reader group" = g$label, "Readers (n)" = length(r),
    "Average AUC without CAD" = res$unaided, "Average AUC with CAD" = res$aided,
    "Effect size (95% CI)" = uminus(res$effect), "P value" = res$p, check.names = FALSE)
}

# Comparisons of the effect size between the reader groups (not "All readers"), named as in
# the Table E6 note: "senior vs early-career cardiothoracic" when the specialty is shared,
# otherwise "senior cardiothoracic vs early-career general/other".
rg <- groups[sapply(groups, function(g) g$label != "All readers")]
comparison_name <- function(a, b) {
  if (a$spec == b$spec) sprintf("%s vs %s %s", a$exp, b$exp, a$spec)
  else sprintf("%s %s vs %s %s", a$exp, a$spec, b$exp, b$spec)
}
pairs <- combn(seq_along(rg), 2, simplify = FALSE)
res_p <- lapply(pairs, function(pr) contrast(rg[[pr[1]]]$readers, rg[[pr[2]]]$readers))
p_holm <- p.adjust(sapply(res_p, `[[`, "p"), method = "holm")
comp_rows <- list()
for (k in seq_along(pairs)) {
  cn <- comparison_name(rg[[pairs[[k]][1]]], rg[[pairs[[k]][2]]])
  res <- res_p[[k]]
  cat(sprintf("  %-60s difference %s  P = %s (Holm %s), ddf = %d\n", cn, res$diff,
              fmt_p(res$p), fmt_p(p_holm[k]), res$df))
  comp_rows[[k]] <- data.frame(
    "Comparison (first minus second group)" = cn, "Difference (95% CI)" = uminus(res$diff),
    "P value" = fmt_p(res$p), "Holm-adjusted P value" = fmt_p(p_holm[k]), check.names = FALSE)
}
om <- omnibus(setNames(lapply(rg, `[[`, "readers"), sapply(rg, `[[`, "label")))
n_words <- c("two", "three", "four")[length(rg) - 1]
cat(sprintf("  Omnibus across %d groups: F(%d, %d) = %.2f, P = %s\n",
            length(rg), om$df1, om$df2, om$F, fmt_p(om$p)))
comp_rows[[length(comp_rows) + 1]] <- data.frame(
  "Comparison (first minus second group)" = sprintf("Across the %s groups", n_words),
  "Difference (95% CI)" = sprintf("F(%d, %d) = %.2f", om$df1, om$df2, om$F),
  "P value" = fmt_p(om$p), "Holm-adjusted P value" = "", check.names = FALSE)

write.csv(do.call(rbind, table_rows), out_path, row.names = FALSE, fileEncoding = "UTF-8")
write.csv(do.call(rbind, comp_rows), comp_path, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("\nWrote %s\nWrote %s\n", out_path, comp_path))
