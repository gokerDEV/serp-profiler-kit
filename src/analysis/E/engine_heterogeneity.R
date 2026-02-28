#!/usr/bin/env Rscript

# Dependencies: install.packages(c("fixest", "data.table", "argparse", "broom"))
# This script implements Analysis E (RQ6): Engine Heterogeneity

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

parser <- ArgumentParser(description = "Analysis E: Engine Heterogeneity (RQ6) (R Implementation)")
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

# Define Variables
metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

core_predictors <- c(semantics, readability, performance, accessibility)

# Standardize Continuous Predictors (for practical_flag threshold comparability)
cols_to_scale <- core_predictors[core_predictors %in% names(df)]
if (length(cols_to_scale) > 0) {
    for (v in cols_to_scale) {
        df[, (v) := as.numeric(scale(df[[v]]))]
    }
    cat("Standardized", length(cols_to_scale), "continuous predictors.\n")
}

# --- RQ6: Pooled Interaction Model ---
# Model: recip_rank ~ (Core Predictors) * search_engine + FE
# fixest syntax for interaction: x1:search_engine or i(search_engine, x1)

cat("Running Pooled Interaction Model...\n")
interaction_terms <- paste0("i(search_engine, ", core_predictors, ", ref='google')")
formula_str <- paste(
    "recip_rank ~",
    paste(core_predictors, collapse = " + "),
    "+",
    paste(interaction_terms, collapse = " + "),
    "+ i(search_engine, ref='google')", # Main effect of engine
    "| search_term"
)
f_interaction <- as.formula(formula_str)

results_list <- list()

tryCatch(
    {
        # Full model
        mod_int_full <- feols(f_interaction, data = df, cluster = ~search_term)

        # Extract
        res_table <- coeftable(mod_int_full)
        res_dt <- as.data.table(res_table, keep.rownames = "term")
        setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

        res_dt[, `:=`(
            model_id = "RQ6_Interaction_R_Full",
            n_obs = nobs(mod_int_full),
            r2 = r2(mod_int_full, "r2"),
            r2_within = r2(mod_int_full, "wr2"),
            software = "R_fixest",
            subset = "Full"
        )]

        ci <- confint(mod_int_full)
        res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

        results_list[["Interaction_Full"]] <- res_dt

        # Split models
        if ("is_source_domain" %in% names(df)) {
            mod_int_split <- feols(f_interaction, data = df, cluster = ~search_term, split = ~is_source_domain)
            for (i in seq_along(mod_int_split)) {
                mod <- mod_int_split[[i]]
                res_tab <- coeftable(mod)
                res_dt_split <- as.data.table(res_tab, keep.rownames = "term")
                setnames(res_dt_split, c("term", "coef", "se", "tstat", "pval"))

                split_val <- names(mod_int_split)[i]
                subset_name <- ifelse(grepl("FALSE", split_val, ignore.case = TRUE), "NoSource",
                    ifelse(grepl("TRUE", split_val, ignore.case = TRUE), "Source", split_val)
                )

                res_dt_split[, `:=`(
                    model_id = paste0("RQ6_Interaction_R_", subset_name),
                    n_obs = nobs(mod),
                    r2 = r2(mod, "r2"),
                    r2_within = r2(mod, "wr2"),
                    software = "R_fixest",
                    subset = subset_name
                )]

                ci_split <- confint(mod)
                res_dt_split[, `:=`(ci_lower = ci_split[, 1], ci_upper = ci_split[, 2])]

                results_list[[paste0("Interaction_", subset_name)]] <- res_dt_split
            }
        }
    },
    error = function(e) {
        cat("Error in Interaction Model:", e$message, "\n")
    }
)

# --- RQ6: Engine-Stratified Models ---
engines <- unique(as.character(df$search_engine))

for (eng in engines) {
    cat("Running Stratified Model for:", eng, "...\n")
    df_sub <- df[search_engine == eng]

    f_strat <- as.formula(paste(
        "recip_rank ~", paste(core_predictors, collapse = " + "), "| search_term"
    ))

    tryCatch(
        {
            mod_strat_full <- feols(f_strat, data = df_sub, cluster = ~search_term)

            res_table <- coeftable(mod_strat_full)
            res_dt <- as.data.table(res_table, keep.rownames = "term")
            setnames(res_dt, c("term", "coef", "se", "tstat", "pval"))

            res_dt[, `:=`(
                model_id = paste0("RQ6_Stratified_", eng, "_R_Full"),
                n_obs = nobs(mod_strat_full),
                r2 = r2(mod_strat_full, "r2"),
                r2_within = r2(mod_strat_full, "wr2"),
                software = "R_fixest",
                subset = paste0("Engine_", eng, "_Full")
            )]

            ci <- confint(mod_strat_full)
            res_dt[, `:=`(ci_lower = ci[, 1], ci_upper = ci[, 2])]

            results_list[[paste0("Stratified_", eng, "_Full")]] <- res_dt

            # Split stratified models
            if ("is_source_domain" %in% names(df_sub) && length(unique(df_sub$is_source_domain)) > 1) {
                mod_strat_split <- feols(f_strat, data = df_sub, cluster = ~search_term, split = ~is_source_domain)
                for (i in seq_along(mod_strat_split)) {
                    mod <- mod_strat_split[[i]]
                    res_tab <- coeftable(mod)
                    res_dt_split <- as.data.table(res_tab, keep.rownames = "term")
                    setnames(res_dt_split, c("term", "coef", "se", "tstat", "pval"))

                    split_val <- names(mod_strat_split)[i]
                    split_name <- ifelse(grepl("FALSE", split_val, ignore.case = TRUE), "NoSource",
                        ifelse(grepl("TRUE", split_val, ignore.case = TRUE), "Source", split_val)
                    )

                    res_dt_split[, `:=`(
                        model_id = paste0("RQ6_Stratified_", eng, "_R_", split_name),
                        n_obs = nobs(mod),
                        r2 = r2(mod, "r2"),
                        r2_within = r2(mod, "wr2"),
                        software = "R_fixest",
                        subset = paste0("Engine_", eng, "_", split_name)
                    )]

                    ci_split <- confint(mod)
                    res_dt_split[, `:=`(ci_lower = ci_split[, 1], ci_upper = ci_split[, 2])]
                    results_list[[paste0("Stratified_", eng, "_", split_name)]] <- res_dt_split
                }
            } else if ("is_source_domain" %in% names(df_sub)) {
                # Only one value exists (either only source or only nosource found)
                split_val <- df_sub$is_source_domain[1]
                split_name <- ifelse(split_val == FALSE, "NoSource", "Source")
                res_dt_copy <- copy(res_dt)
                res_dt_copy[, `:=`(
                    model_id = paste0("RQ6_Stratified_", eng, "_R_", split_name),
                    subset = paste0("Engine_", eng, "_", split_name)
                )]
                results_list[[paste0("Stratified_", eng, "_", split_name)]] <- res_dt_copy
            }
        },
        error = function(e) {
            cat("Error in Stratified Model", eng, ":", e$message, "\n")
        }
    )
}

# Bind and Save
# Bind and Save
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
        practical_flag = (abs(effect_size) >= 0.03) # Standardized threshold assumption
    )]

    out_path <- file.path(args$out_dir, "heterogeneity_coeffs_r.csv")
    fwrite(all_results, out_path)
    cat("Saved R results to", out_path, "\n")
} else {
    cat("No results generated.\n")
}
