#!/usr/bin/env Rscript

# Generate deterministic, R-authoritative AGLM fixtures.
#
# This is deliberately outside EasyGLM's runtime.  It is a compatibility
# harness for the original AGLM 0.4.1 implementation, and records both the
# transformed matrices and fitted glmnet results.  Install into an isolated
# library, for example:
#
#   R CMD INSTALL -l /private/tmp/easyglm-aglm-r-library aglm_0.4.1.tar.gz
#   R_LIBS=/private/tmp/easyglm-aglm-r-library Rscript \
#     scripts/generate_aglm_reference.R tests/fixtures/aglm_0_4_1_reference.json

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: generate_aglm_reference.R OUTPUT.json", call. = FALSE)
}

if (!requireNamespace("aglm", quietly = TRUE)) {
  stop("aglm is required in R_LIBS; do not generate a substitute fixture.", call. = FALSE)
}
if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("glmnet is required in R_LIBS; do not generate a substitute fixture.", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to write the fixture.", call. = FALSE)
}

matrix_rows <- function(x) unname(lapply(seq_len(nrow(x)), function(i) unname(x[i, ])))
named_values <- function(x) as.list(stats::setNames(as.numeric(x), names(x)))

version <- as.character(utils::packageVersion("aglm"))
if (!identical(version, "0.4.1")) {
  stop(sprintf("Expected aglm 0.4.1, found %s.", version), call. = FALSE)
}

# A deliberately uneven grid exposes the normalisation of C dummies.  The
# training data has ties at 0 and 5; the scoring values test both directions
# of default extrapolation.
breaks <- c(0, 2, 5, 10)
x <- c(0, 0, 1, 2, 3, 5, 5, 7, 10, 10)
weights <- c(1.0, 0.7, 1.4, 0.8, 1.9, 0.6, 1.2, 0.9, 1.6, 1.1)
offset <- log(c(0.8, 1.3, 0.7, 1.1, 1.5, 0.9, 1.4, 0.6, 1.2, 1.0))
new_x <- c(-4, 0, 1, 2, 3.5, 5, 7, 10, 14)
new_offset <- log(c(1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 1.0, 0.6))

design_for <- function(values, use_lvar = FALSE, extrapolation = "default") {
  input <- aglm:::newInput(
    data.frame(x = values),
    use_LVar = use_lvar,
    extrapolation = extrapolation,
    bins_list = list(breaks),
    bins_names = list("x")
  )
  aglm:::getDesignMatrix(input)
}

automatic_breaks <- function(values, nbin_max) {
  input <- aglm:::newInput(data.frame(x = values), nbin.max = nbin_max)
  as.numeric(input@vars_info[[1]]$OD_info$breaks)
}

skewed_auto_grid <- c(0, 0, 0, 1, 1, 10, 10, 100, 100, 100)

fit_case <- function(
  name, y, family, alpha, lambda, use_lvar = FALSE,
  extrapolation = "default", fit_x = x, fit_weights = weights,
  fit_offset = offset, control = NULL
) {
  fit <- aglm::aglm(
    data.frame(x = fit_x),
    y,
    family = family,
    alpha = alpha,
    lambda = lambda,
    weights = fit_weights,
    offset = fit_offset,
    use_LVar = use_lvar,
    extrapolation = extrapolation,
    bins_list = list(breaks),
    bins_names = list("x"),
    standardize = TRUE,
    intercept = TRUE,
    control = control
  )
  backend <- fit@backend_models$glmnet
  response <- as.numeric(stats::predict(
    fit,
    newx = data.frame(x = new_x),
    s = lambda,
    type = "response",
    newoffset = new_offset
  ))
  link <- as.numeric(stats::predict(
    fit,
    newx = data.frame(x = new_x),
    s = lambda,
    type = "link",
    newoffset = new_offset
  ))
  coefficients <- as.matrix(stats::coef(backend, s = lambda))
  design <- aglm:::getDesignMatrix(aglm:::newInput(
    data.frame(x = fit_x), use_LVar = use_lvar,
    extrapolation = extrapolation, bins_list = list(breaks),
    bins_names = list("x")
  ))
  normalized_weights <- fit_weights / sum(fit_weights)
  weighted_mean <- colSums(design * normalized_weights)
  weighted_sd <- sqrt(colSums(
    sweep(design, 2, weighted_mean, "-")^2 * normalized_weights
  ))
  list(
    name = name,
    family = family,
    alpha = alpha,
    lambda = lambda,
    numeric_method = if (use_lvar) "L" else "C",
    extrapolation = extrapolation,
    train_x = fit_x,
    weights = fit_weights,
    offset = fit_offset,
    response = y,
    glmnet = list(
      version = as.character(utils::packageVersion("glmnet")),
      standardize = TRUE,
      intercept = TRUE,
      dev_ratio = as.numeric(backend$dev.ratio),
      nulldev = as.numeric(backend$nulldev),
      n_passes = as.integer(backend$npasses),
      coefficients = named_values(stats::setNames(
        as.numeric(coefficients[, 1]), rownames(coefficients)
      ))
    ),
    standardization = list(
      weighted_mean = as.numeric(weighted_mean),
      weighted_sd = as.numeric(weighted_sd)
    ),
    solver_control = control,
    predictions = list(link = link, response = response)
  )
}

fit_cases <- function(control = NULL) {
  list(
    fit_case("poisson_lasso", c(0, 1, 0, 1, 2, 1, 3, 2, 4, 5), "poisson", 1.0, 0.015, control = control),
    fit_case("poisson_elastic", c(0, 1, 0, 1, 2, 1, 3, 2, 4, 5), "poisson", 0.5, 0.015, control = control),
    fit_case("poisson_l_lasso", c(0, 1, 0, 1, 2, 1, 3, 2, 4, 5), "poisson", 1.0, 0.015, use_lvar = TRUE, control = control),
    fit_case("poisson_l_flat_elastic", c(0, 1, 0, 1, 2, 1, 3, 2, 4, 5), "poisson", 0.5, 0.015, use_lvar = TRUE, extrapolation = "flat", control = control),
    fit_case("gaussian_elastic", c(-1.2, -0.9, -0.4, -0.2, 0.1, 0.5, 0.8, 1.1, 1.7, 2.0), "gaussian", 0.5, 0.02, control = control),
    fit_case(
      "binomial_elastic",
      rep(c(0, 0, 0, 0, 1, 0, 1, 1, 1, 1), 2),
      "binomial", 0.5, 0.01,
      fit_x = rep(x, 2), fit_weights = rep(weights, 2),
      fit_offset = rep(offset, 2), control = control
    )
  )
}

fixture <- list(
  schema_version = 1,
  provenance = list(
    generator = "scripts/generate_aglm_reference.R",
    aglm_version = version,
    aglm_source_url = "https://cran.r-project.org/src/contrib/aglm_0.4.1.tar.gz",
    aglm_source_sha256 = "cc6992d51cece85b8a2bc9b5eb79f0fbc93ae86c23f3e19502a1a96fa1900883",
    r_version = R.version.string,
    generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    note = paste(
      "Default AGLM/glmnet fits are retained as diagnostics. Strict-control fits",
      "are the cross-library numerical oracle because glmnet default coordinate",
      "descent may stop early on AGLM's intentionally collinear C basis."
    ),
    comparison_tolerances = list(
      strict_glum_response_atol = 0.0015,
      strict_glum_objective_atol = 1e-7,
      note = paste(
        "The response tolerance covers glmnet coordinate-descent termination",
        "on the Gaussian elastic case; compare the common penalised objective",
        "as well as predictions. No lambda or P2 rescaling is applied."
      )
    )
  ),
  input = list(
    breaks = breaks,
    train_x = x,
    new_x = new_x,
    new_offset = new_offset,
    automatic_breaks_nbin_5 = automatic_breaks(c(0:9), 5),
    automatic_breaks_skewed_nbin_5 = automatic_breaks(skewed_auto_grid, 5),
    default_extrapolation = "default",
    flat_extrapolation = "flat"
  ),
  transforms = list(
    c_default_train = matrix_rows(design_for(x)),
    c_default_new = matrix_rows(design_for(new_x)),
    c_flat_new = matrix_rows(design_for(new_x, extrapolation = "flat")),
    l_default_train = matrix_rows(design_for(x, use_lvar = TRUE)),
    l_default_new = matrix_rows(design_for(new_x, use_lvar = TRUE)),
    l_flat_new = matrix_rows(design_for(
      new_x, use_lvar = TRUE, extrapolation = "flat"
    ))
  ),
  fits_default = fit_cases(),
  fits_strict = fit_cases(control = list(thresh = 1e-15, maxit = 100000))
)

jsonlite::write_json(fixture, args[[1]], auto_unbox = TRUE, digits = 16, pretty = TRUE)
