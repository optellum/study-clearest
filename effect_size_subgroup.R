#!/usr/bin/env Rscript

# ==================================================================
# DBM effect-size analysis by human-factors / AI-attitude subgroup
# ------------------------------------------------------------------
# For each reader human-factors subgroup (specialty, experience,
# certification, prior AI use) and self-reported AI attitude
# (confirmation / automation / aversion bias, trust in LCP), runs the
# Dorfman-Berbaum-Metz (Hillis-modified) MRMC test of the AI effect
# (aided - unaided Wilcoxon AUC) separately for the "yes" and "no"
# levels of the subgroup, and compares the two effect sizes with a
# two-sample z-test on the RRRC treatment-difference estimate and its
# standard error. Effect sizes and per-treatment AUCs are percentages.
#
# The per-level iMRMC inputs are produced upstream by
# create_subgroup_imrmc_files() in MAIN_create_immr_files.ipynb.
#
# Input format:
#   - iMRMC text files, one per subgroup level (e.g.
#     clearest_results_specialty_yes_patient_lom.imrmc): two modalities
#     trt0 = unaided, trt1 = aided; patient-LOM score per reader x case;
#     binary case truth. Read by RJafroc DfReadDataFile (comma-delimited).
#   - ai_attitudes.csv: per-reader attitude flags (confirmation_bias,
#     automation_bias, aversion_bias, trust_in_lcp, use_ai; 1 = yes / 0 = no)
#     and a "participants" id, used only to label the output rows.
# ==================================================================

suppressMessages({
  library(RJafroc)
})
set.seed(123)

mrmc_dir <- "resources/clearest_main_study/mrmc"
ux_dir   <- "resources/clearest_main_study/ux"
out_file <- file.path(ux_dir, "ux_analysis_boot2000.csv")

FOM            <- "Wilcoxon"
method         <- "DBM"
analysisOption <- "RRRC"  # random readers, random cases
n_boots        <- 2000
round_n        <- 2

# Reader subgroups, each with a "yes" / "no" level.
groups <- c("confirmation_bias", "automation_bias", "aversion_bias", "trust_in_lcp",
            "specialty", "experience", "certification", "use_ai")

# Participant lists per attitude level (specialty / experience / certification
# carry no participant list).
ai_attitudes <- read.csv(file.path(ux_dir, "ai_attitudes.csv"), sep = ",", header = TRUE)

# Format an estimate and its CI as "value (lo,hi)", scaled to a percentage.
fmt <- function(est, lo, hi) {
  paste0(round(est * 100, round_n), " (",
         round(lo * 100, round_n), ",", round(hi * 100, round_n), ")")
}

# DBM report (95% RRRC CIs, UtilOutputReport default alpha) for one subgroup level.
subgroup_effect <- function(group, group_id) {
  infile  <- file.path(mrmc_dir, "all",
                       paste0("clearest_results_", group, "_", group_id, "_patient_lom.imrmc"))
  outfile <- file.path(ux_dir,
                       paste0("clearest_results__", group, "_", group_id, "_patient_lom.imrmc"))
  rocData <- DfReadDataFile(infile, format = "iMRMC", delimiter = ",")
  result <- UtilOutputReport(dataset = rocData, ReportFileBaseName = outfile,
                             FOM = FOM, method = method, overWrite = TRUE, nBoots = n_boots)
  fom  <- result$FOM
  rrrc <- result$RRRC
  list(
    # Effect size aided - unaided (trtMeanDiffs stores trt0 - trt1; sign-flipped).
    effect_size = fmt(-fom$trtMeanDiffs["trt0-trt1", ],
                      -rrrc$ciDiffTrt[, "CIUpper"], -rrrc$ciDiffTrt[, "CILower"]),
    aided_trt   = fmt(fom$trtMeans["trt1", ],
                      rrrc$ciAvgRdrEachTrt["trt1", "CILower"], rrrc$ciAvgRdrEachTrt["trt1", "CIUpper"]),
    unaided_trt = fmt(fom$trtMeans["trt0", ],
                      rrrc$ciAvgRdrEachTrt["trt0", "CILower"], rrrc$ciAvgRdrEachTrt["trt0", "CIUpper"]),
    pvalue_rrrc = round(rrrc$FTests["Treatment", "p"], round_n + 1),
    delta       = fom$trtMeanDiffs["trt0-trt1", ],  # trt0 - trt1
    se          = rrrc$ciDiffTrt["trt0-trt1", "StdErr"]
  )
}

# Participants string for a subgroup level ("" for specialty/experience/certification).
participants_for <- function(group, group_id) {
  if (group %in% c("specialty", "experience", "certification")) return("")
  gid <- ifelse(group_id == "yes", 1, 0)
  paste(ai_attitudes[ai_attitudes[[group]] == gid, "participants"], collapse = " ")
}

# Append one output row (write header only on the first row).
write_row <- function(row, first) {
  write.table(row, file = out_file, sep = ",", row.names = FALSE,
              col.names = first, append = !first)
}

dir.create(ux_dir, showWarnings = FALSE, recursive = TRUE)
first <- TRUE

for (group in groups) {
  levels_res <- list()
  for (group_id in c("yes", "no")) {
    res <- subgroup_effect(group, group_id)
    levels_res[[group_id]] <- res
    cat(sprintf("%-18s %-3s: effect %s  aided %s  unaided %s  p=%s\n",
                group, group_id, res$effect_size, res$aided_trt, res$unaided_trt, res$pvalue_rrrc))
    if (group_id == "yes") {
      write_row(data.frame(
        ux_group = paste(group, "yes", sep = "_"),
        participants = participants_for(group, "yes"),
        dataset = "all", effect_size = res$effect_size, pvalue_z = NA,
        aided_trt = res$aided_trt, unaided_trt = res$unaided_trt,
        pvalue_rrrc = res$pvalue_rrrc), first)
      first <- FALSE
    }
  }

  # Two-sample z-test comparing the aided-unaided effect size between levels.
  y <- levels_res[["yes"]]
  n <- levels_res[["no"]]
  z <- (y$delta - n$delta) / sqrt(y$se^2 + n$se^2)
  pvalue_z <- 2 * (1 - pnorm(abs(z)))
  cat(sprintf("  z-test (yes vs no effect size): z=%.3f  p=%.4g\n", z, pvalue_z))

  # Summary row (carries the "no" level values plus the z-test p-value).
  write_row(data.frame(
    ux_group = paste(group, "no", sep = "_"),
    participants = participants_for(group, "no"),
    dataset = "all", effect_size = n$effect_size, pvalue_z = pvalue_z,
    aided_trt = n$aided_trt, unaided_trt = n$unaided_trt,
    pvalue_rrrc = n$pvalue_rrrc), first = FALSE)
}

cat(sprintf("\nWrote results to %s\n", out_file))
