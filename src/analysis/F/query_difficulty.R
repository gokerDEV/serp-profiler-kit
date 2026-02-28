#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse"))
# This script implements Analysis F (RQ6): Query Difficulty

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

parser <- ArgumentParser(description = "Analysis F: Query Difficulty (RQ7) (R Implementation)")
parser$add_argument("--dataset", type = "character", required = TRUE, help = "Path to prepared Parquet dataset")
parser$add_argument("--out-dir", type = "character", required = TRUE, help = "Output directory")
args <- parser$parse_args()

if (!file.exists(args$dataset)) stop(paste("Dataset not found:", args$dataset))

cat("Loading dataset (Parquet):", args$dataset, "\n")
df <- load_analysis_dataset(args$dataset)

if (!dir.exists(args$out_dir)) dir.create(args$out_dir, recursive = TRUE)

# Ensure types
if (!"recip_rank" %in% names(df) && "rank" %in% names(df)) {
    df[, recip_rank := 1.0 / rank]
}

df[, `:=`(
    search_engine = as.factor(search_engine),
    search_term = as.factor(search_term),
    recip_rank = as.numeric(recip_rank)
)]

# Variables
metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

core_predictors <- c(semantics, readability, performance, accessibility)

# Standardize Continuous Predictors
cols_to_scale <- core_predictors[core_predictors %in% names(df)]
if (length(cols_to_scale) > 0) {
    for (v in cols_to_scale) {
        df[, (v) := as.numeric(scale(df[[v]]))]
    }
    cat("Standardized", length(cols_to_scale), "continuous predictors.\n")
}

# Difficulty Bands (Assuming 'difficulty_band' column exists, otherwise created in Python step)
if (!"difficulty_band" %in% names(df)) {
    cat("Warning: 'difficulty_band' column not found. Skipping analysis.\n")
    quit(status = 0)
}

df[, difficulty_band := as.factor(difficulty_band)]

# --- RQ7: Difficulty Band Interaction Model ---
cat("Running Difficulty Interaction Model...\n")

# Formula: recip_rank ~ (Core) * difficulty_band + search_engine | search_term
interaction_terms <- paste0("i(difficulty_band, ", core_predictors, ")")
formula_str <- paste(
    "recip_rank ~",
    paste(interaction_terms, collapse = " + "),
    "+", paste(core_predictors, collapse = " + "), # Main effects
    "+ i(search_engine, ref='brave') | search_term"
)
f_diff <- as.formula(formula_str)

results_list <- list()

tryCatch(
    {
        mod_diff_full <- feols(f_diff, data = df, cluster = ~search_term)
        res_table <- coeftable(mod_diff_full)
        res_dt <- as.data.table(res_table, keep.rownames = "term")
        setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

        res_dt[, `:=`(
            model_id = "RQ7_Difficulty_Interaction_R_Full",
            subset = "Full",
            n_obs = nobs(mod_diff_full),
            r2 = r2(mod_diff_full, "r2"),
            r2_within = r2(mod_diff_full, "wr2"),
            software = "R_fixest"
        )]

        ci <- confint(mod_diff_full)
        res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]
        results_list[["Difficulty_Interaction_Full"]] <- res_dt
    },
    error = function(e) {
        cat("Error in Difficulty Model (Full):", e$message, "\n")
    }
)

if ("is_source_domain" %in% names(df)) {
    tryCatch(
        {
            df_ns <- df[is_source_domain == FALSE]
            mod_diff_ns <- feols(f_diff, data = df_ns, cluster = ~search_term)
            res_table <- coeftable(mod_diff_ns)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

            res_dt[, `:=`(
                model_id = "RQ7_Difficulty_Interaction_R_NoSource",
                subset = "NoSource",
                n_obs = nobs(mod_diff_ns),
                r2 = r2(mod_diff_ns, "r2"),
                r2_within = r2(mod_diff_ns, "wr2"),
                software = "R_fixest"
            )]

            ci <- confint(mod_diff_ns)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]
            results_list[["Difficulty_Interaction_NoSource"]] <- res_dt
        },
        error = function(e) {
            cat("Error in Difficulty Model (NoSource):", e$message, "\n")
        }
    )
}

# Save
if (length(results_list) > 0) {
    all_results <- rbindlist(results_list, fill = TRUE)

    # Standardize (Plan 4.1)
    setnames(all_results,
        old = c("coef", "ci_lower", "ci_upper", "pval"),
        new = c("effect_size", "ci_lower_95", "ci_upper_95", "p_raw"),
        skip_absent = TRUE
    )

    # FDR (per model ID)
    if ("p_raw" %in% names(all_results)) {
        all_results[, p_fdr := p.adjust(p_raw, method = "BH"), by = model_id]
        all_results[, fdr_significant := (p_fdr < 0.05)]
    }

    # Metadata
    all_results[, `:=`(
        model_family = "FE-continuous",
        evidence_tier = "confirmatory",
        practical_flag = (abs(effect_size) >= 0.03)
    )]

    out_path <- file.path(args$out_dir, "difficulty_coeffs_r.csv")
    fwrite(all_results, out_path)
    cat("Saved R results to", out_path, "\n")
} else {
    cat("No results generated.\n")
}
