#!/usr/bin/env Rscript

# ==================================================================
# Reading-time analysis: aided vs unaided (secondary endpoint)
# ------------------------------------------------------------------
# Linear mixed-effects model of per-read reporting time, testing
# whether AI assistance changes reading time. Time is log-transformed
# and modelled with a fixed aided/unaided effect and crossed random
# intercepts for reader and case:
#     log(time) ~ aided + (1 | reader) + (1 | case)
# The aided coefficient is a log time-ratio, so exp(beta) - 1 is the
# geometric-mean percentage change in reading time with AI. Degrees of
# freedom and the p-value use the Satterthwaite approximation (lmerTest).
#
# Input format: CSV (S1_S2_read_times_all.csv) with one row per read:
# participant_id (reader), study_id (case), aided (1 = with CAD, 0 = without),
# and session_time (reporting time in seconds).
# ==================================================================

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)   # Satterthwaite df / p-values for lmer
})

data_path <- "resources/clearest_main_study/mrmc/S1_S2_read_times_all.csv"

# Fit the LMM and summarise the AI effect as a % change in reading time.
# Expected columns: participant_id, study_id, aided (0/1), session_time (s).
analyze_reading_times <- function(df) {
  df <- df %>%
    mutate(
      reader_id = factor(participant_id),
      case_id   = factor(study_id),
      log_time  = log(pmax(session_time, 1e-9))  # guard against zeros
    ) %>%
    drop_na(log_time, aided, reader_id, case_id)

  model <- lmer(log_time ~ aided + (1 | reader_id) + (1 | case_id), data = df)
  coefs <- coef(summary(model))
  beta  <- coefs["aided", "Estimate"]
  p_ai  <- coefs["aided", "Pr(>|t|)"]

  # aided is a log time-ratio -> exp(beta) - 1 is the % change in reading time.
  ci_log <- confint(model, parm = "aided", method = "Wald")
  pct    <- (exp(beta) - 1) * 100
  pct_ci <- (exp(ci_log) - 1) * 100

  cat(sprintf("Reads modelled: %d\n", nrow(df)))
  cat(sprintf("Change in reading time with AI: %.1f%% (95%% CI: %.1f%%, %.1f%%)\n",
              pct, pct_ci[1], pct_ci[2]))
  cat(sprintf("P-value (aided): %.4g\n", p_ai))

  list(model = model, pct_change = pct, pct_ci = pct_ci, p_value = p_ai)
}

df <- read.csv(data_path)
results <- analyze_reading_times(df)
