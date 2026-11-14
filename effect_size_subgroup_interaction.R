#!/usr/bin/env Rscript

# ==================================================================
# Extended DBM analysis: treatment x subgroup interaction
# ------------------------------------------------------------------
# effect_size_subgroup_dbm.R runs an independent DBM significance test
# (aided vs unaided) per subgroup. This script instead pools the
# subgroup levels into a single model and tests whether the AI effect
# (aided - unaided) DIFFERS across the subgroup levels, i.e. whether
# the subgroup is a significant effect modifier.
#
# Method:
#   1. RJafroc::UtilPseudoValues() gives the DBM jackknife pseudovalues
#      [treatment x reader x case]; their mean over cases reproduces
#      each reader's Wilcoxon AUC (the DBM response variable).
#   2. The treatment dimension is collapsed to the per-(reader,case)
#      pseudovalue DIFFERENCE  D = aided - unaided. The treatment x
#      subgroup interaction is the subgroup contrast on D.
#   3. The subgroup flag (from the *_subgroup_* csv written by
#      MAIN_create_immr_files.ipynb) is mapped onto D, then D is averaged
#      over the OTHER axis to the sampling unit of the subgroup factor,
#      and a one-way ANOVA (base aov) of that mean against subgroup gives
#      the interaction F-test:
#         - reader-defined subgroup -> per-reader mean of D
#           (= each reader's AI effect, aided AUC - unaided AUC);
#           ddf = #readers - #levels
#         - case-defined   subgroup -> per-case mean of D
#           (= each case's AI-effect pseudovalue);
#           ddf = #cases - #levels
#      Because the design is fully crossed (every reader reads every
#      case), this marginal-means ANOVA is the exact test on the correct
#      error stratum. It deliberately avoids a full treatment x subgroup
#      mixed model, which silently misattributes a reader-defined factor
#      to the case-level residual when the treatment:reader variance is
#      estimated at zero (singular fit) -> spuriously inflated p-values.
#
# Input format: per-comparison pair of files under mrmc/all/
#   - clearest_results_subgroup_<name>_patient_lom.csv: one row per
#     (reader, case) with Case, Reader, L (truth 0/1), LOM1 (unaided score),
#     LOM2 (aided score), and a "subgroup" label column.
#   - clearest_results_subgroup_<name>_patient_lom.imrmc: the same data in
#     iMRMC format (read by RJafroc DfReadDataFile, comma-delimited).
# ==================================================================

suppressMessages({
  library(RJafroc)
  library(dplyr)
})

mrmc_dir <- "resources/clearest_main_study/mrmc"
folder <- "all"
endpoint_list <- c("_patient_lom")
alpha <- 0.05
round_n <- 4

# (name, type) registry mirrors create_subgroup_imrmc_files() in
# MAIN_create_immr_files.ipynb.
comparisons <- list(
  list(name = "specialty",         type = "reader"),
  list(name = "experience",        type = "reader"),
  list(name = "certification",     type = "reader"),
  list(name = "use_ai",            type = "reader"),
  list(name = "confirmation_bias", type = "reader"),
  list(name = "automation_bias",   type = "reader"),
  list(name = "aversion_bias",     type = "reader"),
  list(name = "trust_in_lcp",      type = "reader"),
  list(name = "nodule_size",       type = "case"),
  list(name = "attenuation",       type = "case"),
  list(name = "spiculation",       type = "case"),
  list(name = "screening",         type = "case"),
  list(name = "thickness",         type = "case")
)

# Wilcoxon / empirical AUC for a single reader-treatment vector.
auc <- function(scores, L) {
  pos <- scores[L == 1]; neg <- scores[L == 0]
  mean(outer(pos, neg, function(a, b) (a > b) + 0.5 * (a == b)))
}

analyse_comparison <- function(name, type, endpoint) {

  base <- file.path(mrmc_dir, folder,
                    paste0("clearest_results_subgroup_", name, endpoint))
  csv <- read.csv(paste0(base, ".csv"), stringsAsFactors = FALSE)
  d <- DfReadDataFile(paste0(base, ".imrmc"), format = "iMRMC", delimiter = ",")

  # DBM pseudovalues: dim [2 treatments, R readers, K cases]
  PV <- UtilPseudoValues(d, FOM = "Wilcoxon")$jkPseudoValues

  # Reconstruct RJafroc's deterministic index order from the csv:
  #   readers : sorted unique Reader id
  #   cases   : normals (L==0, sorted) then abnormals (L==1, sorted)
  readers_sorted <- sort(unique(csv$Reader))
  truth <- csv %>% group_by(Case) %>% summarise(L = first(L), .groups = "drop")
  cases_order <- c(sort(truth$Case[truth$L == 0]), sort(truth$Case[truth$L == 1]))
  stopifnot(dim(PV)[2] == length(readers_sorted),
            dim(PV)[3] == length(cases_order))

  # Verify ordering: per-reader unaided pseudovalue mean must equal that
  # reader's directly computed unaided AUC (guards against any reindexing).
  fom_pv_unaided <- apply(PV[1, , , drop = FALSE], 2, mean)
  fom_direct <- sapply(readers_sorted, function(r) {
    sub <- csv[csv$Reader == r, ]; auc(sub$LOM1, sub$L)
  })
  stopifnot(max(abs(fom_pv_unaided - fom_direct)) < 1e-8)

  # Collapse treatment -> per-(reader,case) pseudovalue difference.
  D <- PV[2, , ] - PV[1, , ]   # aided - unaided, dim [R, K]
  rownames(D) <- as.character(readers_sorted)
  colnames(D) <- cases_order

  # The treatment x subgroup interaction is a contrast on the AI effect
  # along the subgroup axis. Because the design is fully crossed (every
  # reader reads every case), that contrast is tested exactly on the
  # marginal means of D over the OTHER axis, as a one-way ANOVA on the
  # correct sampling unit (readers for reader-defined subgroups, cases
  # for case-defined subgroups). This always uses the right error
  # stratum (ddf = n_units - n_levels) and cannot collapse to the wrong
  # one, unlike automatic stratum selection in a crossed mixed model.
  if (type == "reader") {
    theta <- rowMeans(D)                 # reader AI effect = reader ΔAUC
    sg <- csv %>% group_by(Reader) %>% summarise(subgroup = first(subgroup), .groups = "drop")
    grp <- sg$subgroup[match(readers_sorted, sg$Reader)]
  } else {
    theta <- colMeans(D)                 # per-case AI-effect pseudovalue
    sg <- csv %>% group_by(Case) %>% summarise(subgroup = first(subgroup), .groups = "drop")
    grp <- sg$subgroup[match(cases_order, sg$Case)]
  }
  unit <- data.frame(theta = theta, subgroup = factor(grp))

  fit <- aov(theta ~ subgroup, data = unit)
  an <- summary(fit)[[1]]
  p_int <- an["subgroup", "Pr(>F)"]
  f_int <- an["subgroup", "F value"]
  df_int <- an["Residuals", "Df"]

  # Directly computed per-level AI effect (mean over readers of the
  # within-subset AUC difference) for interpretability.
  levs <- levels(unit$subgroup)
  level_delta <- sapply(levs, function(lv) {
    if (type == "reader") {
      rs <- unique(csv$Reader[csv$subgroup == lv])
      cs <- unique(csv$Case)
    } else {
      rs <- unique(csv$Reader)
      cs <- unique(csv$Case[csv$subgroup == lv])
    }
    deltas <- sapply(rs, function(r) {
      sub <- csv[csv$Reader == r & csv$Case %in% cs, ]
      auc(sub$LOM2, sub$L) - auc(sub$LOM1, sub$L)
    })
    mean(deltas)
  })
  delta_str <- paste(sprintf("%s=%+.4f", levs, level_delta), collapse = "; ")

  list(
    endpoint = sub("^_", "", endpoint),
    comparison = name,
    subgroup_type = type,
    n_levels = length(levs),
    levels = paste(levs, collapse = "|"),
    n_readers = length(readers_sorted),
    n_cases = length(cases_order),
    df_interaction = round(df_int, 2),
    f_interaction = round(f_int, round_n),
    p_interaction = round(p_int, round_n),
    p_interaction_raw = p_int,
    significant = p_int < alpha,
    ai_effect_by_level = delta_str
  )
}

results <- list()
for (endpoint in endpoint_list) {
  for (cmp in comparisons) {
    res <- tryCatch(
      analyse_comparison(cmp$name, cmp$type, endpoint),
      error = function(e) {
        message(sprintf("ERROR [%s %s]: %s", endpoint, cmp$name, conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(res)) {
      results[[paste(endpoint, cmp$name, sep = "_")]] <- res
      print(sprintf("%-10s %-18s (%-6s): p_int=%.4f  F=%.3f  ddf=%.1f  | %s",
                    res$endpoint, res$comparison, res$subgroup_type,
                    res$p_interaction, res$f_interaction, res$df_interaction,
                    res$ai_effect_by_level))
    }
  }
}

out_df <- do.call(rbind, lapply(results, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
out_file <- file.path(mrmc_dir, "dataset_subgroup_interaction_analysis.csv")
write.csv(out_df, out_file, row.names = FALSE)
cat(sprintf("\nWrote %d rows to %s\n", nrow(out_df), out_file))
