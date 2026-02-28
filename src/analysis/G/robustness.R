#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse"))
# This script implements Analysis G (RQ8): Robustness

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

parser <- ArgumentParser(description = "Analysis G: Robustness (RQ8) (R Implementation)")
parser$add_argument("--dataset", type = "character", required = TRUE, help = "Path to prepared Parquet dataset")
parser$add_argument("--out-dir", type = "character", required = TRUE, help = "Output directory")
args <- parser$parse_args()

if (!file.exists(args$dataset)) stop(paste("Dataset not found:", args$dataset))
df <- load_analysis_dataset(args$dataset)
if (!dir.exists(args$out_dir)) dir.create(args$out_dir, recursive = TRUE)

# Ensure types
# Start by checking if 'recip_rank' exists. If not, maybe it needs standardizing?
if (!"recip_rank" %in% names(df) && "rank" %in% names(df)) {
    df[, recip_rank := 1 / rank]
}

if (!"recip_rank" %in% names(df)) {
    stop("Column 'recip_rank' not found in dataset")
}

df[, `:=`(
    search_engine = as.factor(as.character(search_engine)),
    search_term = as.factor(as.character(search_term)),
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

# Base formula
# We need to construct formula string correctly
# recip_rank ~ x1 + ... + i(search_engine, ref='brave') | search_term
# Check existence of predictors in df
valid_predictors <- cols_to_scale
if (length(valid_predictors) == 0) {
    stop("No valid predictors found in dataset")
}

f_base_str <- paste(
    "recip_rank ~", paste(valid_predictors, collapse = " + "),
    "+ i(search_engine, ref='brave') | search_term"
)
f_base <- as.formula(f_base_str)

results_list <- list()

run_robustness_model <- function(data, subset_name, suffix) {
    tryCatch(
        {
            mod <- feols(f_base, data = data, cluster = ~search_term)

            res_table <- coeftable(mod)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

            res_dt[, `:=`(
                model_id = paste0("RQ8_Robustness_", suffix, "_R"),
                n_obs = nobs(mod),
                r2 = r2(mod, "r2"),
                software = "R_fixest",
                subset = subset_name
            )]

            ci <- confint(mod)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

            return(res_dt)
        },
        error = function(e) {
            cat("Error in", suffix, ":", e$message, "\n")
            return(NULL)
        }
    )
}

# 1. Full Data (Baseline)
cat("Running Baseline...\n")
results_list[["Baseline"]] <- run_robustness_model(df, "Full", "Baseline")

# 2. No Source Domain (if exists)
if ("is_source_domain" %in% names(df)) {
    cat("Running No-Source subset...\n")
    df_no_source <- df[is_source_domain == FALSE]
    results_list[["NoSource"]] <- run_robustness_model(df_no_source, "NoSource", "NoSourceDomain")
}

# 3. Winsorized (Mockup - assumes columns might be winsorized in caller or we do simple cap)
# Doing simple 1-99% winsorization here for R robustness
winsorize <- function(x, prob = 0.01) {
    if (length(x) == 0) {
        return(x)
    }
    q <- quantile(x, probs = c(prob, 1 - prob), na.rm = TRUE)
    x[x < q[1]] <- q[1]
    x[x > q[2]] <- q[2]
    return(x)
}

df_wins <- copy(df)
cols_to_win <- c(readability, performance, accessibility)
for (col in cols_to_win) {
    if (col %in% names(df_wins)) {
        df_wins[[col]] <- winsorize(df_wins[[col]])
    }
}

cat("Running Winsorized...\n")
results_list[["Winsorized"]] <- run_robustness_model(df_wins, "Full_Winsorized", "Winsorized")

# 4. Two-Way Clustering (search_term + domain)
if ("domain" %in% names(df)) {
    cat("Running Two-Way Clustering (search_term + domain)...\n")
    tryCatch(
        {
            # Ensure domain is clean
            df_2way <- df[!is.na(domain)]

            # fixest uses `cluster = c("id1", "id2")` for multi-way
            mod_2way <- feols(f_base, data = df_2way, cluster = c("search_term", "domain"))

            res_table <- coeftable(mod_2way)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

            res_dt[, `:=`(
                model_id = "RQ8_Robustness_2WayCluster_R",
                n_obs = nobs(mod_2way),
                r2 = r2(mod_2way, "r2"),
                software = "R_fixest",
                subset = "Full"
            )]

            ci <- confint(mod_2way)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

            results_list[["TwoWayCluster"]] <- res_dt
        },
        error = function(e) {
            cat("Error in TwoWayCluster:", e$message, "\n")
        }
    )
}


if (length(results_list) > 0) {
    all_results <- rbindlist(results_list, fill = TRUE)

    # Standardize
    setnames(all_results,
        old = c("coef", "ci_lower", "ci_upper", "pval"),
        new = c("effect_size", "ci_lower_95", "ci_upper_95", "p_raw"),
        skip_absent = TRUE
    )

    # Robustness checks usually don't need FDR, but we can do it.
    if ("p_raw" %in% names(all_results)) {
        all_results[, p_fdr := p.adjust(p_raw, method = "BH"), by = model_id]
        all_results[, fdr_significant := (p_fdr < 0.05)]
    }

    # Metadata
    all_results[, `:=`(
        model_family = "FE-continuous",
        evidence_tier = "robustness",
        practical_flag = (abs(effect_size) >= 0.03)
    )]

    out_path <- file.path(args$out_dir, "robustness_coeffs_r.csv")
    fwrite(all_results, out_path)
    cat("Saved R results to", out_path, "\n")
}
