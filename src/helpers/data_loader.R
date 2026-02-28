library(yaml)

get_schema_path <- function(relative_path = "src/analysis_v1.yml") {
    # Resolve from absolute or current execution path
    roots <- c(".", "..", "../..", "../../..", "../../../..")
    for (root in roots) {
        path <- file.path(root, relative_path)
        if (file.exists(path)) {
            return(path)
        }
    }
    stop("Schema file not found")
}

load_analysis_dataset <- function(dataset_path, schema_path = "src/analysis_v1.yml") {
    if (!requireNamespace("arrow", quietly = TRUE)) {
        stop("arrow package is required to read parquet datasets.")
    }

    utils::globalVariables(c(":="))

    cat("Loading dataset processing standardisation from", dataset_path, "\n")
    df <- data.table::as.data.table(arrow::read_parquet(dataset_path))

    multipliers <- get_feature_multipliers(schema_path)
    for (feat in names(multipliers)) {
        mult <- multipliers[[feat]]
        if (feat %in% names(df) && mult != 1) {
            df[, (feat) := get(feat) * mult]
        }
    }

    return(df)
}

get_analysis_categories <- function(schema_path = "src/analysis_v1.yml") {
    real_path <- get_schema_path(schema_path)
    schema <- yaml::read_yaml(real_path)

    analysis <- schema$analysis
    if (is.null(analysis)) {
        return(character(0))
    }
    return(names(analysis))
}

get_analysis_features <- function(category = NULL, schema_path = "src/analysis_v1.yml") {
    real_path <- get_schema_path(schema_path)
    schema <- yaml::read_yaml(real_path)

    analysis <- schema$analysis
    if (is.null(analysis)) {
        return(character(0))
    }

    if (!is.null(category)) {
        if (!is.null(analysis[[category]])) {
            return(as.character(names(analysis[[category]])))
        }
        return(character(0))
    }

    feats <- c()
    for (cat in names(analysis)) {
        if (!is.null(analysis[[cat]])) {
            feats <- c(feats, as.character(names(analysis[[cat]])))
        }
    }
    return(feats)
}

load_analysis_metrics <- function(schema_path = "src/analysis_v1.yml") {
    real_path <- get_schema_path(schema_path)
    schema <- yaml::read_yaml(real_path)

    analysis <- schema$analysis
    res <- list()
    if (!is.null(analysis)) {
        for (cat in names(analysis)) {
            if (!is.null(analysis[[cat]])) {
                res[[cat]] <- as.character(names(analysis[[cat]]))
            } else {
                res[[cat]] <- character(0)
            }
        }
    }
    return(res)
}

get_feature_multipliers <- function(schema_path = "src/analysis_v1.yml") {
    real_path <- get_schema_path(schema_path)
    schema <- yaml::read_yaml(real_path)

    analysis <- schema$analysis
    multipliers <- list()

    if (!is.null(analysis)) {
        for (cat in names(analysis)) {
            features <- analysis[[cat]]
            if (!is.null(features) && is.list(features)) {
                for (feat in names(features)) {
                    mult <- features[[feat]]$multiplier
                    if (!is.null(mult)) {
                        multipliers[[feat]] <- as.numeric(mult)
                    } else {
                        multipliers[[feat]] <- 1
                    }
                }
            }
        }
    }
    return(multipliers)
}

get_rank_tiers <- function(schema_path = "src/analysis_v1.yml") {
    real_path <- get_schema_path(schema_path)
    schema <- yaml::read_yaml(real_path)

    tiers <- schema$rank_tiers
    if (is.null(tiers) || is.null(tiers$bins) || is.null(tiers$labels) || is.null(tiers$logit_cut_points)) {
        stop(paste0("'rank_tiers' configuration with 'bins', 'labels' and 'logit_cut_points' must be defined in ", schema_path))
    }

    return(list(
        bins = as.numeric(tiers$bins),
        labels = as.character(tiers$labels),
        logit_cut_points = as.numeric(tiers$logit_cut_points)
    ))
}
