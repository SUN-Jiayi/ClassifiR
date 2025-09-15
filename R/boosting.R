#' Train an AdaBoost Classifier for Binary Classification
#'
#' This function implements the AdaBoost (Adaptive Boosting) algorithm using decision stumps (one-level decision trees)
#' as weak learners. It is designed for binary classification tasks with numeric features and binary labels (0 or 1).
#' The function trains multiple weak learners sequentially, where each subsequent learner focuses more on instances that
#' were misclassified by previous learners.
#'
#' @param data A data frame containing the training data. All columns except the last are treated as features
#' (must be numeric), and the last column is treated as the binary label (must be 0 or 1).
#' @param n_iter An integer specifying the number of boosting iterations (i.e., number of weak learners to train).
#' Must be a positive count. Default is 50.
#'
#' @details
#' The AdaBoost algorithm assigns a weight to each training instance. Initially, all weights are equal.
#' At each iteration, a decision stump is trained to minimize the weighted classification error.
#' The model assigns a weight (alpha) to each stump based on its accuracy.
#' Then, the weights of the training samples are updated: misclassified samples receive higher weights to focus
#' the next weak learner on harder cases.
#'
#' The weak learners used here are decision stumps, which find the optimal threshold for a single feature to
#' split the data. The algorithm searches over all features and thresholds to find the one with the lowest
#' weighted classification error.
#'
#' The function internally converts labels from \code{0, 1} to \code{-1, 1} as required by the AdaBoost algorithm.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{\code{models}}{A list of trained decision stumps. Each stump is a list containing:
#'     \code{feature} (column index),
#'     \code{threshold} (split point),
#'     \code{polarity} (direction of comparison, either 1 or -1).}
#'   \item{\code{alphas}}{A numeric vector of weights (alpha values) assigned to each weak learner based on its accuracy.}
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate simple binary data
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
#' y <- ifelse(X$x1 + X$x2 + rnorm(100) > 0, 1, 0)
#' data <- cbind(X, y)
#'
#' # Train AdaBoost model
#' model <- adaboost_(data, n_iter = 20)
#' print(model)
#' }
#'
#' @seealso \code{\link{predict_adaboost_}}, \code{\link{tune_adaboost_}}
#'
#' @export
adaboost_ <- function(data, n_iter = 50) {
  assertthat::assert_that(is.data.frame(data), msg = "'data' must be a data frame.")
  assertthat::assert_that(ncol(data) >= 2, msg = "Input 'data' must contain at least one feature and one label column.")
  assertthat::assert_that(assertthat::is.count(n_iter), n_iter > 0,
                          msg = "'n_iter' must be a positive integer.")
  assertthat::assert_that(!anyNA(data), msg = "Input 'data' contains missing values. Please remove or impute them.")
  assertthat::assert_that(
    all(sapply(data[, -ncol(data)], is.numeric)),
    msg = "All feature columns must be numeric."
  )
  X <- data[, -ncol(data)]
  y <- data[[ncol(data)]]
  assertthat::assert_that(all(y %in% c(0, 1)), msg = "Label column must contain only 0 and 1 values.")

  y <- ifelse(y == 0, -1, 1)  # convert to -1, 1
  n <- nrow(X)
  assertthat::assert_that(n > 0, msg = "Training data is empty.")

  w <- rep(1/n, n)
  models <- list()
  alphas <- c()

  for (iter in 1:n_iter) {
    train_stump <- function(X, y, w) {
      best_feature <- NULL
      best_threshold <- NULL
      best_polarity <- 1
      min_error <- Inf

      for (j in seq_len(ncol(X))) {
        feature_values <- X[[j]]
        thresholds <- unique(feature_values)

        for (t in thresholds) {
          for (polarity in c(1, -1)) {
            pred <- ifelse(polarity * feature_values < polarity * t, -1, 1)
            err <- sum(w * (pred != y))
            if (err < min_error) {
              min_error <- err
              best_feature <- j
              best_threshold <- t
              best_polarity <- polarity
            }
          }
        }
      }

      return(list(feature = best_feature, threshold = best_threshold, polarity = best_polarity))
    }

    predict_stump <- function(stump, X) {
      feature_values <- X[[stump$feature]]
      ifelse(stump$polarity * feature_values < stump$polarity * stump$threshold, -1, 1)
    }

    stump <- train_stump(X, y, w)
    pred <- predict_stump(stump, X)
    err <- sum(w * (pred != y))

    if (err >= 0.5 || err == 0 || is.na(err)) {
      next
    }

    alpha <- 0.5 * log((1 - err) / err)
    alphas <- c(alphas, alpha)
    models[[length(models) + 1]] <- stump

    w <- w * exp(-alpha * y * pred)
    w <- w / sum(w)
  }

  return(list(models = models, alphas = alphas))
}


#' Make Predictions Using an AdaBoost Model
#'
#' This function uses a trained AdaBoost model (produced by \code{\link{adaboost_}}) to make binary class predictions on new data.
#' The prediction is performed by aggregating the weighted votes of each decision stump (weak learner) in the model.
#' Each weak learner predicts either \code{-1} or \code{1}, and the final prediction is based on the sign of the weighted sum.
#'
#' @param model A list representing the AdaBoost model, as returned by the \code{\link{adaboost_}} function.
#'   It must contain the following elements:
#'   \describe{
#'     \item{\code{models}}{A list of decision stumps, where each stump is a list with components:
#'       \code{feature} (column index), \code{threshold} (numeric split point), and \code{polarity} (either 1 or -1).}
#'     \item{\code{alphas}}{A numeric vector of weights (alpha values) corresponding to the accuracy of each weak learner.}
#'   }
#' @param newdata A data frame of numeric features on which to make predictions. The feature columns must match those used during training (i.e., same order and number of columns as in the training data).
#'
#' @details
#' The prediction process works as follows:
#' \enumerate{
#'   \item Each weak learner (decision stump) makes a prediction of -1 or 1 on the new data based on its threshold and polarity.
#'   \item Each prediction is weighted by the corresponding alpha value (model confidence).
#'   \item The weighted predictions are summed for each row.
#'   \item The final predicted label is 1 if the sum is non-negative, and 0 otherwise.
#' }
#'
#' This implementation assumes that the AdaBoost model was trained on data where the binary labels were originally \code{0} and \code{1}, and internally converted to \code{-1} and \code{1}.
#' The function converts the final result back to 0/1 format for user-friendly output.
#'
#' If the model contains no weak learners (i.e., \code{model$models} is empty), a warning is issued and a vector of 0s is returned.
#'
#' @return An integer vector of predicted class labels (0 or 1), one for each row in \code{newdata}.
#'
#' @examples
#' \dontrun{
#' # Train an AdaBoost model
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
#' y <- ifelse(X$x1 + X$x2 > 0, 1, 0)
#' model <- adaboost_(cbind(X, y), n_iter = 10)
#'
#' # Predict on new data
#' new_X <- data.frame(x1 = rnorm(5), x2 = rnorm(5))
#' preds <- predict_adaboost_(model, new_X)
#' print(preds)
#' }
#'
#' @seealso \code{\link{adaboost_}}, \code{\link{tune_adaboost_}}
#'
#' @export
predict_adaboost_ <- function(model, newdata) {
  assertthat::assert_that(
    is.list(model),
    !is.null(model$models),
    !is.null(model$alphas),
    msg = "Invalid model format."
  )

  assertthat::assert_that(
    is.data.frame(newdata),
    !anyNA(newdata),
    msg = "'newdata' must be a data frame without missing values."
  )

  assertthat::assert_that(
    is.list(model$models),
    all(sapply(model$models, function(stump) {
      is.list(stump) &&
        all(c("feature", "threshold", "polarity") %in% names(stump))
    })),
    msg = "Each stump in model$models must be a list with 'feature', 'threshold', and 'polarity'."
  )

  assertthat::assert_that(
    is.numeric(model$alphas),
    length(model$alphas) == length(model$models),
    !anyNA(model$alphas),
    msg = "model$alphas must be a numeric vector matching model$models and contain no NA."
  )

  if (length(model$models) == 0) {
    return(rep(0, nrow(newdata)))
  }

  num_features <- max(sapply(model$models, function(stump) stump$feature))
  assertthat::assert_that(ncol(newdata) >= num_features,
              msg = sprintf("New data has fewer columns (%d) than required (%d).",
                            ncol(newdata), num_features))

  final_score <- rep(0, nrow(newdata))

  for (i in seq_along(model$models)) {
    stump <- model$models[[i]]
    pred <- {
      feature_values <- newdata[[stump$feature]]
      ifelse(stump$polarity * feature_values < stump$polarity * stump$threshold, -1, 1)
    }
    final_score <- final_score + model$alphas[[i]] * pred
  }

  return(as.integer(ifelse(final_score >= 0, 1, 0)))
}

#' Tune AdaBoost Hyperparameters via Grid Search on Number of Iterations
#'
#' This function automatically tunes the number of boosting iterations (\code{n_iter}) for an AdaBoost classifier
#' by evaluating different values on a provided validation set. It returns the model with the highest validation accuracy.
#'
#' @param train_data A data frame containing the training set, where the last column is the binary target variable (0 or 1),
#'   and the remaining columns are numeric features.
#' @param val_data A data frame containing the validation set, structured the same way as \code{train_data}, with the last column
#'   as the binary target variable (0 or 1), and the rest as numeric features.
#' @param iter_options A numeric vector of candidate values for \code{n_iter}, representing the number of boosting rounds
#'   (i.e., the number of weak learners to include in the final model). Must be a vector of positive integers. Default is \code{c(10, 20, 30, 50)}.
#' @param verbose Logical. If \code{TRUE}, the function will print the accuracy corresponding to each \code{n_iter} tried. Useful for monitoring progress.
#'
#' @details
#' The function performs a grid search over the values in \code{iter_options}. For each candidate number of iterations:
#' \enumerate{
#'   \item An AdaBoost model is trained using \code{\link{adaboost_}} on the \code{train_data}.
#'   \item Predictions are made on the \code{val_data} using \code{\link{predict_adaboost_}}.
#'   \item The prediction accuracy on the validation set is computed.
#'   \item The model with the highest validation accuracy is retained and returned at the end.
#' }
#'
#' If multiple values achieve the same highest accuracy, the one with the smallest \code{n_iter} will be returned (since we iterate in order).
#' If all models fail to train (e.g., due to empty weak learner lists), the function throws an error.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{\code{model}}{The trained AdaBoost model (in the same format as returned by \code{\link{adaboost_}}) that achieved the best accuracy.}
#'   \item{\code{n_iter}}{The value of \code{n_iter} (number of boosting rounds) that gave the best validation performance.}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data
#' set.seed(123)
#' train_data <- data.frame(
#'   x1 = rnorm(100),
#'   x2 = rnorm(100),
#'   y = ifelse(runif(100) > 0.5, 1, 0)
#' )
#' val_data <- data.frame(
#'   x1 = rnorm(50),
#'   x2 = rnorm(50),
#'   y = ifelse(runif(50) > 0.5, 1, 0)
#' )
#'
#' # Tune n_iter values
#' result <- tune_adaboost_(
#'   train_data,
#'   val_data,
#'   iter_options = c(5, 10, 15),
#'   verbose = TRUE
#' )
#'
#' print(result$n_iter)
#' }
#'
#' @seealso \code{\link{adaboost_}}, \code{\link{predict_adaboost_}}
#'
#' @export
tune_adaboost_ <- function(train_data, val_data, iter_options = c(10, 20, 30, 50), verbose = FALSE) {
  assertthat::assert_that(is.data.frame(train_data), is.data.frame(val_data),
              msg = "Both 'train_data' and 'val_data' must be data frames.")
  assertthat::assert_that(nrow(train_data) > 0, nrow(val_data) > 0,
              msg = "Training or validation data is empty.")
  assertthat::assert_that(ncol(train_data) == ncol(val_data),
              msg = "Train and validation data must have the same number of columns.")
  assertthat::assert_that(is.numeric(iter_options), all(iter_options %% 1 == 0),
                          all(iter_options > 0), msg = "'iter_options' must be a vector of positive integers.")
  X <- train_data[ , -ncol(train_data)]
  X_val <- val_data[ , -ncol(val_data)]
  assertthat::assert_that(all(sapply(X, is.numeric)),
                          msg = "All feature columns of training set must be numeric.")
  assertthat::assert_that(all(sapply(X_val, is.numeric)),
                          msg = "All feature columns of validation set must be numeric.")
  y <- train_data[[ncol(train_data)]]
  assertthat::assert_that(all(y %in% c(0, 1)),
                          msg = "Target variable must be binary (0 or 1).")
  y_val <- val_data[[ncol(val_data)]]
  assertthat::assert_that(all(y_val %in% c(0, 1)),
                          msg = "Validation target variable must be binary (0 or 1).")

  best_acc <- -Inf
  best_model <- NULL
  best_iter <- NULL

  for (n_iter in iter_options) {
    model <- adaboost_(train_data, n_iter = n_iter)
    if (length(model$models) == 0) {
      next
    }

    pred <- predict_adaboost_(model, val_data)
    true <- val_data[[ncol(val_data)]]
    acc <- mean(pred == true)

    if (verbose) {
      cat(sprintf("n_iter = %d -> Accuracy: %.4f\n", n_iter, acc))
    }

    if (acc > best_acc) {
      best_acc <- acc
      best_model <- model
      best_iter <- n_iter
    }
  }

  assertthat::assert_that(!is.null(best_model), msg = "All models failed during tuning. Check data or parameters.")

  return(list(model = best_model, n_iter = best_iter))
}


# data("weatherAUS")
# # data("train")
# # data("test")
# clean <- dataPreprocessing(weatherAUS, label_col = "RainTomorrow")
# train_test_list <- splitDataset_(clean)
# train <- train_test_list$train
# test <- train_test_list$test
# is.data.frame(train)
# # Split the data into train and validation
# # set.seed(42)
# idx <- sample(nrow(train), 0.8 * nrow(train))
# train_part <- train[idx, ]
# val_part <- train[-idx, ]
# model <- adaboost_(data = train, n_iter = 30)
# pred <- predict_adaboost_(model, test)
# true <- test[, 91]
# acc <- mean(pred == true)
# acc
#
# # Automatic hyper parameter tuning
# result <- tune_adaboost_(train_part, val_part, iter_options = c(10, 20, 30, 40, 50))
# model <- result$model
#
# # Predict using the optimal model
# pred <- predict_adaboost_(model, test)
# true <- test[, 95]
# acc <- mean(pred == true)
# cat("Final test accuracy with best model:", acc, "\n")
