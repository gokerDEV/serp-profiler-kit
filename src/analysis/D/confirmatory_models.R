#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse", "broom"))
# This script implements the Analysis D Confirmatory Models using R's 'fixest' for rigorous FE estimation.

suppressPackageStartupMessages({
    library(fixest)
    library(data.table)
    library(argparse)
    library(arrow)
})

# Find and source data_loader.R
source_helper <- function(relative_path) {
    paths <- c(relative_path, file.path("..", relative_path), file.path("../..", relative_path), file.path("../../..", relative_path))
    for (p in paths) {
        if (file.exists(p)) {
            source(p)
            return(invisible())
        }
    }
    stop(paste("Could not find helper:", relative_path))
}

invisible(source_helper("src/helpers/data_loader.R"))

parser <- ArgumentParser(description = "Analysis D: Confirmatory Models (R Implementation)")
parser$add_argument("--dataset", type = "character", required = TRUE, help = "Path to prepared Parquet dataset")
parser$add_argument("--out-dir", type = "character", required = TRUE, help = "Output directory")
args <- parser$parse_args()

# Load Data
if (!file.exists(args$dataset)) {
    stop(paste("Dataset not found:", args$dataset))
}

cat("Loading dataset (Parquet):", args$dataset, "\n")
df <- load_analysis_dataset(args$dataset)

# Ensure output directory
if (!dir.exists(args$out_dir)) {
    dir.create(args$out_dir, recursive = TRUE)
}

# Ensure types
# Check for search_engine column existence
if (!"search_engine" %in% names(df)) {
    stop("Error: 'search_engine' column missing from dataset.")
}

# Ensure all potential predictor columns are numeric
metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

cols_to_check <- c(semantics, readability, performance, accessibility)

for (col in cols_to_check) {
    if (col %in% names(df)) {
        # Convert to numeric, handle potential character conversion issues
        if (!is.numeric(df[[col]])) {
            df[[col]] <- as.numeric(as.character(df[[col]]))
        }
    }
}

if (!"recip_rank" %in% names(df) && "rank" %in% names(df)) {
    df[, recip_rank := 1.0 / rank]
}

df[, `:=`(
    search_engine = as.factor(search_engine),
    search_term = as.factor(search_term),
    recip_rank = as.numeric(as.character(recip_rank))
)]

# Define Model Formulas (align with Python)
# ---- Confirmatory Models ----
# RQ2: Semantics
f_rq2_str <- paste("recip_rank ~", paste(semantics, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term")
f_rq2 <- as.formula(f_rq2_str)

# RQ3: Readability
f_rq3_str <- paste("recip_rank ~", paste(c(semantics, readability), collapse = " + "), "+ i(search_engine, ref = 'google') | search_term")
f_rq3 <- as.formula(f_rq3_str)

# RQ4: Performance (Standardized)
f_rq4_str <- paste("recip_rank ~", paste(c(semantics, performance), collapse = " + "), "+ i(search_engine, ref = 'google') | search_term")
f_rq4 <- as.formula(f_rq4_str)

# RQ5: Accessibility (Standardized)
f_rq5_str <- paste("recip_rank ~", paste(c(semantics, accessibility), collapse = " + "), "+ i(search_engine, ref = 'google') | search_term")
f_rq5 <- as.formula(f_rq5_str)

confirmatory_models <- list(
    RQ2_Semantics = f_rq2,
    RQ3_Readability = f_rq3,
    RQ4_Performance = f_rq4,
    RQ5_Accessibility = f_rq5
)

results_list_conf <- list()

# Helper function to run/extract models
run_model_set <- function(model_list, results_container, model_type_prefix) {
    for (m_name in names(model_list)) {
        cat("Running", m_name, "...\n")

        # Prepare Data based on Model Requirements
        # Required standardizing for Performance & Accessibility
        needs_standardization <- any(grepl("Performance|Accessibility", m_name))

        if (needs_standardization) {
            df_model <- copy(df)
            cols_to_scale <- c(semantics, performance, accessibility)
            cols_present <- intersect(cols_to_scale, names(df_model))

            if (length(cols_present) > 0) {
                safe_scale <- function(x) {
                    if (!is.numeric(x)) {
                        return(rep(NA, length(x)))
                    }
                    s <- sd(x, na.rm = TRUE)
                    if (is.na(s) || s == 0) {
                        return(rep(0, length(x)))
                    }
                    return(as.numeric(scale(x)))
                }
                for (v in cols_present) {
                    df_model[, (v) := safe_scale(df_model[[v]])]
                }
                cat("  Standardized variables.\n")
            }
        } else {
            df_model <- df
        }

        tryCatch(
            {
                # Full model
                mod_full <- feols(model_list[[m_name]], data = df_model, cluster = ~search_term)
                res_table_full <- coeftable(mod_full)
                res_dt_full <- as.data.table(res_table_full, keep.rownames = "term")
                setnames(res_dt_full, c("term", "coef", "se", "tstat", "pval"))

                res_dt_full[, `:=`(
                    model_id = paste0(m_name, "_R_FE"),
                    subset = "Full",
                    n_obs = nobs(mod_full),
                    r2 = r2(mod_full, "r2"),
                    r2_within = r2(mod_full, "wr2"),
                    software = "R_fixest"
                )]

                ci_full <- confint(mod_full)
                res_dt_full[, `:=`(ci_lower = ci_full[, 1], ci_upper = ci_full[, 2])]

                results_container[[paste0(m_name, "_Full")]] <- res_dt_full

                # Split model if is_source_domain exists
                if ("is_source_domain" %in% names(df_model)) {
                    mod_multi <- feols(model_list[[m_name]], data = df_model, cluster = ~search_term, split = ~is_source_domain)

                    # Ensure it is a list of models to iterate over
                    for (i in seq_along(mod_multi)) {
                        mod <- mod_multi[[i]]
                        res_table <- coeftable(mod)
                        res_dt <- as.data.table(res_table, keep.rownames = "term")
                        setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

                        split_val <- names(mod_multi)[i]
                        # Fixest names the splits with the variable value. Example: "FALSE" or "TRUE".
                        subset_name <- ifelse(grepl("FALSE", split_val, ignore.case = TRUE), "NoSource",
                            ifelse(grepl("TRUE", split_val, ignore.case = TRUE), "Source", split_val)
                        )

                        res_dt[, `:=`(
                            model_id = paste0(m_name, "_R_FE"),
                            subset = subset_name,
                            n_obs = nobs(mod),
                            r2 = r2(mod, "r2"),
                            r2_within = r2(mod, "wr2"),
                            software = "R_fixest"
                        )]

                        ci <- confint(mod)
                        res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

                        results_container[[paste0(m_name, "_", subset_name)]] <- res_dt
                    }
                }
            },
            error = function(e) {
                cat("Error in", m_name, ":", e$message, "\n")
            }
        )
    }
    return(results_container)
}

results_list_conf <- run_model_set(confirmatory_models, results_list_conf, "Confirmatory")

# Bind, format and save
format_and_save <- function(results_list, out_filename, evidence_tier_val) {
    if (length(results_list) == 0) {
        return()
    }

    all_results <- rbindlist(results_list, fill = TRUE)
    setnames(all_results,
        old = c("coef", "ci_lower", "ci_upper", "pval"),
        new = c("effect_size", "ci_lower_95", "ci_upper_95", "p_raw"),
        skip_absent = TRUE
    )

    all_results[, p_fdr := p.adjust(p_raw, method = "BH"), by = model_id]
    all_results[, `:=`(
        model_family = "FE-continuous",
        evidence_tier = evidence_tier_val
    )]
    all_results[, practical_flag := abs(effect_size) >= 0.03]

    out_path <- file.path(args$out_dir, out_filename)
    fwrite(all_results, out_path)
    cat("Saved R", evidence_tier_val, "results to", out_path, "\n")
}

format_and_save(results_list_conf, "confirmatory_coeffs_r.csv", "confirmatory")
