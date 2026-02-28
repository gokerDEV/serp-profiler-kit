#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(fixest)
    library(data.table)
    library(argparse)
    library(arrow)
})

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

parser <- ArgumentParser()
parser$add_argument("--dataset", type = "character", required = TRUE)
parser$add_argument("--out-dir", type = "character", required = TRUE)
args <- parser$parse_args()

df <- load_analysis_dataset(args$dataset)
if (!dir.exists(args$out_dir)) dir.create(args$out_dir, recursive = TRUE)

metrics <- load_analysis_metrics()
semantics <- metrics$semantic
readability <- metrics$readability
performance <- metrics$performance
accessibility <- metrics$accessibility

cols_to_check <- c(semantics, readability, performance, accessibility)
for (col in cols_to_check) {
    if (col %in% names(df) && !is.numeric(df[[col]])) {
        df[[col]] <- as.numeric(as.character(df[[col]]))
    }
}
if (!"recip_rank" %in% names(df) && "rank" %in% names(df)) df[, recip_rank := 1.0 / rank]

df[, `:=`(
    search_engine = as.factor(search_engine),
    search_term = as.factor(search_term),
    recip_rank = as.numeric(as.character(recip_rank))
)]

# Standardization for all to ensure model likelihoods compare identical setups identically
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
for (v in cols_to_check) {
    if (v %in% names(df)) df[, (v) := safe_scale(df[[v]])]
}

# Run models sequentially, dropping missing values carefully
vars_needed <- c("recip_rank", "search_engine", "search_term", semantics, readability, performance, accessibility)
# But missingness varies by feature. To do valid nested comparisons, the dataset must be exactly identical.
# I will use df_complete for each comparison individually.

results <- list()

compare_nested <- function(name, model_name, added_block, vars_base, vars_full) {
    needed <- unique(c("recip_rank", "search_engine", "search_term", vars_full))
    df_sub <- na.omit(df[, needed, with = FALSE])

    f_base <- as.formula(paste("recip_rank ~", paste(vars_base, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term"))
    if (length(vars_base) == 0) {
        f_base <- as.formula("recip_rank ~ i(search_engine, ref = 'google') | search_term")
    }
    f_full <- as.formula(paste("recip_rank ~", paste(vars_full, collapse = " + "), "+ i(search_engine, ref = 'google') | search_term"))

    m_base <- feols(f_base, data = df_sub, cluster = ~search_term)
    m_full <- feols(f_full, data = df_sub, cluster = ~search_term)

    ll_base <- logLik(m_base)
    ll_full <- logLik(m_full)

    lr_stat <- 2 * as.numeric(ll_full - ll_base)
    df_diff <- length(coef(m_full)) - length(coef(m_base))
    p_val <- pchisq(lr_stat, df = df_diff, lower.tail = FALSE)

    r2_base <- r2(m_base, "r2")
    r2_full <- r2(m_full, "r2")
    delta_r2 <- r2_full - r2_base

    delta_aic <- AIC(m_full) - AIC(m_base)
    delta_bic <- BIC(m_full) - BIC(m_base)

    results[[length(results) + 1]] <<- data.table(
        Model = model_name,
        Added_Block = added_block,
        Delta_R2 = delta_r2,
        Delta_AIC = delta_aic,
        Delta_BIC = delta_bic,
        LR_stat = lr_stat,
        LR_p = p_val,
        n_obs = nrow(df_sub)
    )
}

# 1. Semantics vs Baseline
compare_nested("Semantics", "RQ2_Semantics", "Semantics", character(0), semantics)

# 2. Readability vs Semantics
compare_nested("Readability", "RQ3_Readability", "Readability", semantics, c(semantics, readability))

# 3. Performance vs Semantics
compare_nested("Performance", "RQ4_Performance", "Performance", semantics, c(semantics, performance))

# 4. Accessibility vs Semantics
compare_nested("Accessibility", "RQ5_Accessibility", "Accessibility", semantics, c(semantics, accessibility))

res_df <- rbindlist(results)
fwrite(res_df, file.path(args$out_dir, "nested_model_fit.csv"))
cat("Saved nested model fit to nested_model_fit.csv\n")
