#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse", "broom"))
# This script implements the Analysis D Supplementary Models using R's 'fixest'.

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

parser <- ArgumentParser(description = "Analysis D: Supplementary Models (R Implementation)")
parser$add_argument("--dataset", type = "character", required = TRUE, help = "Path to prepared Parquet dataset")
parser$add_argument("--out-dir", type = "character", required = TRUE, help = "Output directory")
args <- parser$parse_args()

if (!file.exists(args$dataset)) stop(paste("Dataset not found:", args$dataset))

cat("Loading dataset (Parquet):", args$dataset, "\n")
df <- load_analysis_dataset(args$dataset)
if (!dir.exists(args$out_dir)) dir.create(args$out_dir, recursive = TRUE)

metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

for (col in c(semantics, readability, performance, accessibility)) {
    if (col %in% names(df) && !is.numeric(df[[col]])) df[[col]] <- as.numeric(as.character(df[[col]]))
}
if (!"recip_rank" %in% names(df) && "rank" %in% names(df)) df[, recip_rank := 1.0 / rank]

df[, `:=`(search_engine = as.factor(search_engine), search_term = as.factor(search_term), recip_rank = as.numeric(as.character(recip_rank)))]

# Define Model Formulas
supplementary_models <- list()
if (length(readability) > 0) supplementary_models[["Supp_Readability"]] <- as.formula(paste("recip_rank ~", paste(readability, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term"))
if (length(performance) > 0) supplementary_models[["Supp_Performance"]] <- as.formula(paste("recip_rank ~", paste(performance, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term"))
if (length(accessibility) > 0) supplementary_models[["Supp_Accessibility"]] <- as.formula(paste("recip_rank ~", paste(accessibility, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term"))

results_list_supp <- list()

for (m_name in names(supplementary_models)) {
    cat("Running", m_name, "...\n")
    needs_standardization <- any(grepl("Readability|Performance|Accessibility", m_name))
    if (needs_standardization) {
        df_model <- copy(df)
        cols_to_scale <- c(readability, performance, accessibility)
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
            for (v in cols_present) df_model[, (v) := safe_scale(df_model[[v]])]
        }
    } else {
        df_model <- df
    }
    tryCatch(
        {
            mod_full <- feols(supplementary_models[[m_name]], data = df_model, cluster = ~search_term)
            res_table <- coeftable(mod_full)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))
            res_dt[, `:=`(model_id = paste0(m_name, "_R_FE_Full"), n_obs = nobs(mod_full), r2 = r2(mod_full, "r2"), r2_within = r2(mod_full, "wr2"), software = "R_fixest", subset = "Full")]
            ci <- confint(mod_full)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]
            results_list_supp[[paste0(m_name, "_Full")]] <- res_dt

            # Split models
            if ("is_source_domain" %in% names(df_model) && length(unique(df_model$is_source_domain)) > 1) {
                mod_split <- feols(supplementary_models[[m_name]], data = df_model, cluster = ~search_term, split = ~is_source_domain)
                for (i in seq_along(mod_split)) {
                    mod <- mod_split[[i]]
                    res_tab <- coeftable(mod)
                    res_dt_split <- as.data.table(res_tab, keep.rownames = "term")
                    setnames(res_dt_split, c("term", "coef", "se", "tstat", "pval"))

                    split_val <- names(mod_split)[i]
                    subset_name <- ifelse(grepl("FALSE", split_val, ignore.case = TRUE), "NoSource",
                        ifelse(grepl("TRUE", split_val, ignore.case = TRUE), "Source", split_val)
                    )

                    res_dt_split[, `:=`(
                        model_id = paste0(m_name, "_R_FE_", subset_name),
                        n_obs = nobs(mod),
                        r2 = r2(mod, "r2"),
                        r2_within = r2(mod, "wr2"),
                        software = "R_fixest",
                        subset = subset_name
                    )]

                    ci_split <- confint(mod)
                    res_dt_split[, `:=`(ci_lower = ci_split[, 1], ci_upper = ci_split[, 2])]

                    results_list_supp[[paste0(m_name, "_", subset_name)]] <- res_dt_split
                }
            }
        },
        error = function(e) cat("Error in", m_name, ":", e$message, "\n")
    )
}

if (length(results_list_supp) > 0) {
    all_results <- rbindlist(results_list_supp, fill = TRUE)
    setnames(all_results, old = c("coef", "ci_lower", "ci_upper", "pval"), new = c("effect_size", "ci_lower_95", "ci_upper_95", "p_raw"), skip_absent = TRUE)
    all_results[, p_fdr := p.adjust(p_raw, method = "BH"), by = model_id]
    all_results[, `:=`(model_family = "FE-continuous", evidence_tier = "supplementary")]
    all_results[, practical_flag := abs(effect_size) >= 0.03]
    out_path <- file.path(args$out_dir, "supplementary_coeffs_r.csv")
    fwrite(all_results, out_path)
    cat("Saved R supplementary results to", out_path, "\n")
}
