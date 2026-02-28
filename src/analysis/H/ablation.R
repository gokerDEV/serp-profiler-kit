#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse"))
# This script implements Analysis H (RQ8): Ablation (Inferential)

suppressPackageStartupMessages({
    library(fixest)
    library(data.table)
    library(argparse)
    library(arrow)
    library(yaml)
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

parser <- ArgumentParser(description = "Analysis H: Ablation (RQ9) (R Implementation)")
parser$add_argument("--dataset", type = "character", required = TRUE, help = "Path to prepared Parquet dataset")
parser$add_argument("--out-dir", type = "character", required = TRUE, help = "Output directory")
args <- parser$parse_args()

if (!file.exists(args$dataset)) stop(paste("Dataset not found:", args$dataset))
df <- load_analysis_dataset(args$dataset)
if (!dir.exists(args$out_dir)) dir.create(args$out_dir, recursive = TRUE)

# Ensure types
# Start by checking if 'recip_rank' exists. If not, maybe it needs standardizing from rank?
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
# Load Schema Features
metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

models <- list(
    "M0_Base" = c(),
    "M1_Semantics" = semantics,
    "M2_Sem_Read" = c(semantics, readability),
    "M3_Sem_Read_Perf" = c(semantics, readability, performance),
    "M4_Full" = c(semantics, readability, performance, accessibility)
)

core_predictors <- c(semantics, readability, performance, accessibility)

# Standardize Continuous Predictors (for practical_flag threshold comparability)
cols_to_scale <- core_predictors[core_predictors %in% names(df)]
if (length(cols_to_scale) > 0) {
    for (v in cols_to_scale) {
        df[, (v) := as.numeric(scale(df[[v]]))]
    }
    cat("Standardized", length(cols_to_scale), "continuous predictors.\n")
}

results_list <- list()

cat("Running Ablation Models...\n")

for (m_name in names(models)) {
    predictors <- models[[m_name]]

    if (length(predictors) == 0) {
        f_str <- "recip_rank ~ i(search_engine, ref='brave') | search_term"
    } else {
        f_str <- paste(
            "recip_rank ~", paste(predictors, collapse = " + "),
            "+ i(search_engine, ref='brave') | search_term"
        )
    }

    tryCatch(
        {
            mod <- feols(as.formula(f_str), data = df, cluster = ~search_term)

            # Calculate AIC: n * log(SSR/n) + 2k
            n <- nobs(mod)
            ssr <- sum(resid(mod)^2)
            k <- length(coef(mod)) # Fixed effects not counted in k for AIC usually in large N, or handled by fixest?
            # fixest::AIC(mod) is available
            aic_val <- AIC(mod)

            # Extract Coeffs
            res_table <- coeftable(mod)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

            res_dt[, `:=`(
                model_id = paste0("RQ9_", m_name, "_R"),
                n_obs = n,
                r2 = r2(mod, "r2"),
                aic = aic_val,
                software = "R_fixest"
            )]

            ci <- confint(mod)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

            results_list[[m_name]] <- res_dt
        },
        error = function(e) {
            cat("Error in", m_name, ":", e$message, "\n")
        }
    )
}

# --- Stability Metrics (Coefficient % Change) ---
# Compare M1 vs M2, M2 vs M3 for shared coefficients
stability_list <- list()

# Helper to get coeffs as vector
get_coefs <- function(res_dt) {
    vec <- res_dt$coef
    names(vec) <- res_dt$term
    return(vec)
}

comparisons <- list(
    c("M1_Semantics", "M2_Sem_Read"),
    c("M2_Sem_Read", "M3_Sem_Read_Perf"),
    c("M3_Sem_Read_Perf", "M4_Full")
)

for (pair in comparisons) {
    m_prev <- pair[1]
    m_curr <- pair[2]

    if (m_prev %in% names(results_list) && m_curr %in% names(results_list)) {
        c_prev <- get_coefs(results_list[[m_prev]])
        c_curr <- get_coefs(results_list[[m_curr]])

        shared_terms <- intersect(names(c_prev), names(c_curr))
        # Ignore intercept, FE dumps? Term names usually clean "sim_title", etc.

        if (length(shared_terms) > 0) {
            # Calculate % change
            changes <- abs((c_curr[shared_terms] - c_prev[shared_terms]) / c_prev[shared_terms])
            avg_change <- mean(changes, na.rm = TRUE) * 100

            # Find significant shifts (>20%)
            shifts <- changes[changes > 0.2]
            shift_vars <- names(shifts)

            stability_list[[paste0(m_prev, "_vs_", m_curr)]] <- data.table(
                comparison = paste0(m_prev, "_vs_", m_curr),
                avg_pct_change = avg_change,
                unstable_count = length(shift_vars),
                unstable_vars = paste(shift_vars, collapse = ";")
            )
        }
    }
}


if (length(results_list) > 0) {
    all_results <- rbindlist(results_list, fill = TRUE)

    # Standardize
    setnames(all_results,
        old = c("coef", "ci_lower", "ci_upper", "pval"),
        new = c("effect_size", "ci_lower_95", "ci_upper_95", "p_raw"),
        skip_absent = TRUE
    )

    if ("p_raw" %in% names(all_results)) {
        all_results[, p_fdr := p.adjust(p_raw, method = "BH"), by = model_id]
        all_results[, fdr_significant := (p_fdr < 0.05)]
    }

    # Metadata
    all_results[, `:=`(
        model_family = "FE-continuous",
        evidence_tier = "ablation_inferential",
        practical_flag = (abs(effect_size) >= 0.03)
    )]

    out_path <- file.path(args$out_dir, "ablation_inferential_r.csv")
    fwrite(all_results, out_path)
    cat("Saved R results to", out_path, "\n")

    if (length(stability_list) > 0) {
        stab_res <- rbindlist(stability_list)
        out_stab <- file.path(args$out_dir, "ablation_stability_r.csv")
        fwrite(stab_res, out_stab)
        cat("Saved Stability results to", out_stab, "\n")
    }
}
