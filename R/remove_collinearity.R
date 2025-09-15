#' Detect and Optionally Remove Multicollinearity Based on VIF
#'
#' This function calculates the Variance Inflation Factor (VIF) for each numeric column
#' in a data frame to assess multicollinearity among predictors. Optionally, it can iteratively
#' remove the most collinear variables based on a user-defined threshold.
#'
#'
#' @param data A data frame containing only numeric columns. Non-numeric columns will trigger an error.
#' @param threshold A numeric value greater than 1. VIF values above this threshold indicate multicollinearity.
#'                  If \code{remove = TRUE}, variables with the highest VIF above this threshold will be removed iteratively.
#'                  Default is 5.
#' @param remove Logical. If \code{TRUE}, variables with VIF above the threshold will be iteratively removed.
#'               If \code{FALSE}, only VIF values will be returned without modifying the input data. Default is \code{FALSE}.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{data}{The cleaned data frame with collinear variables removed (if \code{remove = TRUE}).}
#'   \item{vif}{A named numeric vector containing VIF values for the remaining variables.}
#'   \item{removed}{A character vector of variable names that were removed due to high multicollinearity. Empty if \code{remove = FALSE}.}
#' }
#'
#' @details
#' The function uses the definition of VIF as \eqn{1 / (1 - R^2)} where \eqn{R^2} is obtained
#' by regressing each variable on all others. A VIF of 1 indicates no multicollinearity;
#' higher values indicate increasing levels of collinearity.
#'
#' Perfect or near-perfect multicollinearity (i.e., \eqn{R^2} close to 1) will trigger a warning
#' and the VIF is set to \code{Inf}. If a model fails to fit (e.g., due to singularities), the VIF
#' is set to \code{NA} and a warning is issued.
#'
#' @section Warnings:
#' \itemize{
#'   \item If the model for a variable fails to fit, a warning is shown and its VIF is set to \code{NA}.
#'   \item If perfect or near-perfect multicollinearity is detected, a warning is issued and VIF is set to \code{Inf}.
#'   \item If fewer than two variables remain after removal, a warning is shown.
#' }
#'
#' @examples
#' # Create a data frame with collinear variables
#' set.seed(123)
#' df <- data.frame(
#'   x1 = rnorm(100)
#' )
#' df$x2 <- df$x1 + rnorm(100, sd = 0.01)  # Highly collinear with x1
#' df$x3 <- rnorm(100)  # Not collinear
#'
#' # Process collinearity with removal
#' result <- process_collinearity_(df, threshold = 5, remove = TRUE)
#' result$vif         # View VIF values
#' result$removed     # Variables removed due to high VIF
#'
#' # Only compute VIFs without removing variables
#' process_collinearity_(df, threshold = 5, remove = FALSE)
#'
#' @import assertthat
#' @export
process_collinearity_ <- function(data, threshold = 5, remove = FALSE) {
  assertthat::assert_that(is.data.frame(data), msg = "'data' must be a data frame")
  assertthat::assert_that(all(sapply(data, is.numeric)), msg = "All columns should be numeric")
  assertthat::assert_that(!any(is.na(data)), msg = "data frame cannot contain missing values")
  assertthat::assert_that(is.numeric(threshold), threshold > 1, msg = "threshold must be a numeric value larger than 1")
  assertthat::assert_that(is.logical(remove), msg = "remove should be either TRUE or FALSE")
  assertthat::assert_that(ncol(data) > 1, msg = "At least two numeric columns are required to compute VIF.")
  non_numeric_cols <- names(data)[!sapply(data, is.numeric)]
  assertthat::assert_that(length(non_numeric_cols) == 0,
                          msg = paste("Non-numeric columns detected:", paste(non_numeric_cols, collapse = ", ")))

  compute_vif <- function(df) {
    vif_values <- numeric(ncol(df))
    names(vif_values) <- colnames(df)
    for (i in seq_along(df)) {
      target <- df[[i]]
      others <- df[, -i, drop = FALSE]
      model <- try(lm(target ~ ., data = others), silent = TRUE)
      if (inherits(model, "try-error")) {
        vif_values[i] <- NA
      } else {
        summary_model <- suppressWarnings(summary(model))
        r2 <- summary_model$r.squared
        vif_values[i] <- if (is.na(r2) || r2 >= 1) Inf else 1 / (1 - r2)
      }
    }
    return(vif_values)
  }

  data_clean <- data
  removed_cols <- c()

  repeat {
    vif_vals <- compute_vif(data_clean)

    if (!remove) {
      break  # only calculate vif and not delete the columns
    }

    max_vif <- max(vif_vals, na.rm = TRUE)
    if (max_vif <= threshold) {
      break  # all variables satisfy the requirements
    }

    # Find the column with highest vif, record and delete it.
    col_to_remove <- names(which.max(vif_vals))
    removed_cols <- c(removed_cols, col_to_remove)
    data_clean <- data_clean[, !(names(data_clean) == col_to_remove), drop = FALSE]
    if (ncol(data_clean) < 2) {
      break
    }
  }

  # recalculate final VIF
  final_vif <- compute_vif(data_clean)

  return(list(
    data = data_clean,
    vif = final_vif,
    removed = removed_cols
  ))
}
# data("weatherAUS")
# w <- dataPreprocessing(weatherAUS, label_col = "RainTomorrow")
# x <- calculate_vif(w, 5, remove = TRUE)
# x
# df <- data.frame(
#   x1 = rnorm(100)
# )
# df$x2 <- df$x1 + 0.0001
# df$x3 <- df$x1 + 0.0002
# result <- process_collinearity_(df, threshold = 2, remove = TRUE)
