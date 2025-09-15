#' Logistic Regression Classifier with Optional Hyperparameter Tuning
#'
#' Fits a logistic regression model using gradient descent for binary classification.
#' Optionally performs internal grid search over learning rate and number of iterations
#' using a validation split from the training data.
#'
#' @param train_data A numeric data frame. The last column must be the binary response variable (0 or 1).
#' @param test_data A numeric data frame. The last column must be the binary response variable (0 or 1).
#' @param learning_rates A numeric vector of candidate learning rates to tune from. Default is \code{c(0.001, 0.01, 0.1)}.
#' @param n_iters A numeric vector of candidate iteration counts to tune from. Default is \code{c(500, 1000)}.
#' @param tune Logical. If \code{TRUE}, performs internal hyperparameter tuning. If \code{FALSE}, uses first value from \code{learning_rates} and \code{n_iters}.
#'
#' @import assertthat
#' @seealso \code{\link[assertthat]{assert_that}}
#'
#' @return A list containing:
#' \item{theta}{The trained model coefficients.}
#' \item{predict_fun}{A function to predict class labels and probabilities.}
#' \item{prediction_train}{Predicted class labels for the full training data.}
#' \item{probability_train}{Predicted probabilities for the full training data.}
#' \item{prediction}{Predicted class labels for the test data.}
#' \item{probability}{Predicted probabilities for the test data.}
#' \item{best_learning_rate}{The learning rate chosen during tuning (or manually set).}
#' \item{best_n_iter}{The number of iterations chosen during tuning (or manually set).}
#' @examples
#' # Create a simple binary dataset
#' train_data <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 0, 1, 0), y = c(0, 0, 1, 1))
#' test_data <- data.frame(x1 = c(1.5, 2.5), x2 = c(0.5, 1), y = c(0, 1))
#'
#' # Run logistic regression
#' model <- my_logistic_regression(train_data, test_data)
#' model$prediction
#' model$probability
#' @export

my_logistic_regression <- function(train_data, test_data,
                                   learning_rates = c(0.001, 0.01, 0.1),
                                   n_iters = c(500, 1000),
                                   tune = TRUE) {
  # Assertions
  assertthat::assert_that(nrow(train_data) > 0, msg = "Training data must not be empty.")
  assertthat::assert_that(nrow(test_data) > 0, msg = "Test data must not be empty.")
  assertthat::assert_that(!any(is.na(train_data)), msg = "Training data contains missing values.")
  assertthat::assert_that(!any(is.na(test_data)), msg = "Test data contains missing values.")

  # Split data into X and y
  X_train_all <- as.matrix(train_data[, -ncol(train_data)])
  y_train_all <- train_data[[ncol(train_data)]]
  X_test <- as.matrix(test_data[, -ncol(test_data)])
  y_test <- test_data[[ncol(test_data)]]

  assertthat::assert_that(all(y_train_all %in% c(0, 1)), msg = "Target variable must be binary (0 or 1).")
  assertthat::assert_that(all(y_test %in% c(0, 1)), msg = "Target variable must be binary (0 or 1).")

  # Normalize
  normalize <- function(x) (x - min(x)) / (max(x) - min(x))
  X_train_all <- apply(X_train_all, 2, normalize)
  X_test <- apply(X_test, 2, normalize)

  # Optional: parameter tuning using 80/20 split
  if (tune) {
    set.seed(123)
    idx <- sample(1:nrow(X_train_all), size = 0.8 * nrow(X_train_all))
    X_train <- X_train_all[idx, , drop = FALSE]
    y_train <- y_train_all[idx]
    X_val <- X_train_all[-idx, , drop = FALSE]
    y_val <- y_train_all[-idx]

    best_acc <- -Inf
    for (lr in learning_rates) {
      for (iter in n_iters) {
        theta <- rep(0, ncol(X_train) + 1)
        X <- cbind(1, X_train)
        sigmoid <- function(z) 1 / (1 + exp(-z))
        for (i in 1:iter) {
          z <- X %*% theta
          h <- sigmoid(z)
          gradient <- t(X) %*% (h - y_train) / nrow(X_train)
          theta <- theta - lr * gradient
        }
        preds <- ifelse(sigmoid(cbind(1, X_val) %*% theta) >= 0.5, 1, 0)
        acc <- mean(preds == y_val)
        if (acc > best_acc) {
          best_acc <- acc
          best_theta <- theta
          best_lr <- lr
          best_iter <- iter
        }
      }
    }
    cat(sprintf("Best learning_rate = %.3f, n_iter = %d, validation accuracy = %.3f",
                    best_lr, best_iter, best_acc))
    theta <- best_theta
  } else {
    X_train <- X_train_all
    y_train <- y_train_all
    X <- cbind(1, X_train)
    theta <- rep(0, ncol(X))
    sigmoid <- function(z) 1 / (1 + exp(-z))
    for (i in 1:n_iter) {
      z <- X %*% theta
      h <- sigmoid(z)
      gradient <- t(X) %*% (h - y_train) / nrow(X_train)
      theta <- theta - learning_rate * gradient
    }
  }

  # Prediction function
  predict <- function(X_new) {
    X_new <- cbind(1, X_new)
    probs <- 1 / (1 + exp(-X_new %*% theta))
    preds <- as.numeric(probs >= 0.5)
    list(probs = probs, preds = preds)
  }

  train_output <- predict(X_train_all)
  test_output <- predict(X_test)

  return(list(
    theta = theta,
    predict_fun = predict,
    prediction_train = train_output$preds,
    probability_train = train_output$probs,
    prediction = test_output$preds,
    probability = test_output$probs,
    best_learning_rate = if (tune) best_lr else learning_rate,
    best_n_iter = if (tune) best_iter else n_iter
  ))
}
