#' Linear SVM Classifier (from scratch) with Hyperparameter Tuning
#'
#' Trains a linear SVM classifier using stochastic gradient descent with optional tuning
#' for both learning rate and regularization strength (lambda).
#'
#' @param train_data A data frame. Last column is the binary label (0 or 1).
#' @param test_data A data frame. Last column is the binary label.
#' @param learning_rates A numeric vector of learning rates to try (e.g., c(0.001, 0.01)). Default is 0.01.
#' @param lambda_values A numeric vector of lambda (regularization) values to try. Default is c(0.0001, 0.001, 0.01, 0.1).
#' @param n_iter Number of training iterations. Default is 1000.
#' @param tune Logical. If TRUE, performs tuning. Default is TRUE.
#'
#' @import assertthat
#' @seealso \code{\link[assertthat]{assert_that}}
#'
#' @return A list containing:
#' \item{weights}{The final trained weights.}
#' \item{predict_fun}{Prediction function.}
#' \item{prediction_train}{Predicted classes for training data.}
#' \item{probability_train}{Predicted probabilities for training data.}
#' \item{prediction}{Predicted classes for test data.}
#' \item{probability}{Predicted probabilities for test data.}
#' \item{best_lambda}{The selected lambda.}
#' \item{best_learning_rate}{The selected learning rate.}
#' @examples
#' # Create simple binary dataset
#' train_data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(1, 1, 0, 0), y = c(0, 0, 1, 1))
#' test_data <- data.frame(x1 = c(2, 3), x2 = c(1, 0), y = c(0, 1))
#'
#' # Train SVM with tuning disabled
#' model <- my_svm(train_data, test_data, learning_rate = 0.01, n_iter = 10, tune = FALSE)
#' model$prediction
#' model$probability
#' @export
my_svm <- function(train_data, test_data,
                   learning_rates = c(0.01),
                   lambda_values = c(0.0001, 0.001, 0.01, 0.1),
                   n_iter = 1000,
                   tune = TRUE) {

  # Checks
  assertthat::assert_that(!any(is.na(train_data)), msg = "Training data has NAs")
  assertthat::assert_that(!any(is.na(test_data)), msg = "Test data has NAs")
  assertthat::assert_that(nrow(train_data) > 0, msg = "Training data is empty.")
  assertthat::assert_that(nrow(test_data) > 0, msg = "Test data is empty.")

  # Data split
  X_all <- as.matrix(train_data[, -ncol(train_data)])
  y_all <- train_data[[ncol(train_data)]]
  X_test <- as.matrix(test_data[, -ncol(test_data)])
  assertthat::assert_that(all(y_all %in% c(0, 1)))
  y_all_svm <- ifelse(y_all == 0, -1, 1)

  # Normalize
  normalize <- function(x) (x - min(x)) / (max(x) - min(x))
  X_all <- apply(X_all, 2, normalize)
  X_test <- apply(X_test, 2, normalize)

  # Validation split
  set.seed(42)
  idx <- sample(1:nrow(X_all), 0.8 * nrow(X_all))
  X_train <- X_all[idx, ]
  y_train <- y_all_svm[idx]
  X_val <- X_all[-idx, ]
  y_val <- y_all[-idx]

  best_lambda <- lambda_values[1]
  best_lr <- learning_rates[1]
  best_acc <- -Inf

  # Training function
  train_model <- function(X, y, lambda, learning_rate) {
    X <- cbind(1, X)
    w <- rep(0, ncol(X))
    for (epoch in 1:n_iter) {
      for (i in 1:nrow(X)) {
        margin <- y[i] * sum(w * X[i, ])
        if (margin < 1) {
          w <- w + learning_rate * (y[i] * X[i, ] - 2 * lambda * w)
        } else {
          w <- w + learning_rate * (-2 * lambda * w)
        }
      }
    }
    return(w)
  }

  # Grid search for best (lambda, learning_rate)
  if (tune) {
    for (lambda_try in lambda_values) {
      for (lr_try in learning_rates) {
        w_try <- train_model(X_train, y_train, lambda_try, lr_try)
        val_margin <- cbind(1, X_val) %*% w_try
        val_pred <- ifelse(val_margin >= 0, 1, 0)
        acc <- mean(val_pred == y_val)

        if (acc > best_acc) {
          best_acc <- acc
          best_lambda <- lambda_try
          best_lr <- lr_try
        }
      }
    }

    cat(sprintf("Best learning_rate = %.4f, best lambda = %.4f, validation accuracy = %.4f",
                    best_lr, best_lambda, best_acc))
  }

  # Final training on full training set
  w <- train_model(X_all, y_all_svm, best_lambda, best_lr)

  # Prediction function
  predict_fun <- function(X_new) {
    X_new <- cbind(1, X_new)
    margins <- X_new %*% w
    probs <- 1 / (1 + exp(-margins))
    preds <- ifelse(margins >= 0, 1, 0)
    return(list(preds = preds, probs = probs, margins = margins))
  }

  train_output <- predict_fun(X_all)
  test_output <- predict_fun(X_test)

  return(list(
    weights = w,
    predict_fun = predict_fun,
    prediction_train = train_output$preds,
    probability_train = train_output$probs,
    prediction = test_output$preds,
    probability = test_output$probs,
    best_lambda = best_lambda,
    best_learning_rate = best_lr
  ))
}
