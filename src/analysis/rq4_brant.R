#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
infile  <- args[1]
outfile <- args[2]

suppressMessages({
  library(MASS)
  library(brant)
  library(jsonlite)
})

# Veri oku
df <- read.csv(infile, stringsAsFactors = FALSE)

# Hedefi ordered factor yap
df$serp_quintile <- factor(df$serp_quintile, ordered = TRUE)

# Tüm diğer sütunları predictor olarak kullan
form <- as.formula("serp_quintile ~ .")

# Ordinal logit
model <- polr(form, data = df, method = "logistic", Hess = TRUE)

# Brant testi (gerçek)
bt <- brant(model)   # bt: class 'brant', altında tablo var

# Çıktıyı tabloya çevir
res_tab <- as.data.frame(bt)

# Omnibus satırını bul
if (!"Omnibus" %in% rownames(res_tab)) {
  stop("Could not find 'Omnibus' row in Brant output")
}

omni <- res_tab["Omnibus", , drop = FALSE]

chi2_val <- as.numeric(omni[1, "X2"])
df_val   <- as.integer(omni[1, "df"])
p_val    <- as.numeric(omni[1, "probability"])

out <- list(
  chi2_statistic     = chi2_val,
  degrees_of_freedom = df_val,
  p_value            = p_val
)

write(toJSON(out, auto_unbox = TRUE), file = outfile)
