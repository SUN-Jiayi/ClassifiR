#' K-Nearest Neighbors Classifier with Optional k Tuning
#'
#' Predicts binary labels using the K-Nearest Neighbors algorithm.
#' Optionally selects the best \code{k} by internal validation from training data.
#'
#' @param train_data A data frame or matrix. Last column is the binary label (0 or 1).
#' @param test_data A data frame or matrix. Last column is the binary label (not used in prediction).
#' @param k_values A vector of candidate values for \code{k}. Default is \code{c(3, 5, 7)}.
#' @param tune Logical. If \code{TRUE}, performs internal evaluation to choose the best \code{k}.
#'
#' @import assertthat
#' @seealso \code{\link[assertthat]{assert_that}}
#'
#' @return A list with:
#' \item{prediction}{Predicted class labels (0 or 1) for the test set.}
#' \item{probability}{Proportion of neighbors that voted for class 1 (for test set).}
#' \item{best_k}{The value of \code{k} used for final prediction.}
#'
#' @examples
#' # Create a simple dataset
#' train_data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(4, 3, 2, 1), y = c(0, 0, 1, 1))
#' test_data <- data.frame(x1 = c(2.5, 3.5), x2 = c(2.5, 1.5), y = c(0, 1))
#'
#' # Run KNN
#' model <- my_knn(train_data, test_data, k = 3)
#' model$prediction
#' model$probability
#'
#' @export

my_knn <- function(train_data, test_data, k_values = c(3, 5, 7), tune = FALSE) {
  assertthat::assert_that(nrow(train_data) > 0, msg = "Training data must not be empty.")
  assertthat::assert_that(nrow(test_data) > 0, msg = "Test data must not be empty.")
  assertthat::assert_that(!any(is.na(train_data)), msg = "Training data contains missing values.")
  assertthat::assert_that(!any(is.na(test_data)), msg = "Test data contains missing values.")

  assertthat::assert_that(all(sapply(train_data[, -ncol(train_data)], is.numeric)),
                          msg = "All training features must be numeric.")
  assertthat::assert_that(all(sapply(test_data[, -ncol(test_data)], is.numeric)),
                          msg = "All test features must be numeric.")

  X_all <- as.matrix(train_data[, -ncol(train_data)])
  y_all <- train_data[[ncol(train_data)]]
  y_test <- test_data[[ncol(test_data)]]

  assertthat::assert_that(all(y_all %in% c(0, 1)), msg = "Training labels must be binary (0 or 1).")

  normalize <- function(x) (x - min(x)) / (max(x) - min(x))
  X_all <- apply(X_all, 2, normalize)
  X_test <- apply(as.matrix(test_data[, -ncol(test_data)]), 2, normalize)

  # Split into internal train/val sets (80/20)
  set.seed(42)
  idx <- sample(1:nrow(X_all), 0.8 * nrow(X_all))
  X_train <- X_all[idx, ]
  y_train <- y_all[idx]
  X_val <- X_all[-idx, ]
  y_val <- y_all[-idx]

  # Tune k if needed
  best_k <- k_values[1]
  best_acc <- -Inf

  if (tune) {
    for (k_try in k_values) {
      preds <- numeric(nrow(X_val))
      for (i in 1:nrow(X_val)) {
        dists <- sqrt(rowSums((t(t(X_train) - X_val[i, ]))^2))
        neighbor_idx <- order(dists)[1:min(k_try, length(dists))]
        preds[i] <- ifelse(mean(y_train[neighbor_idx]) >= 0.5, 1, 0)
      }
      acc <- mean(preds == y_val)
      if (acc > best_acc) {
        best_acc <- acc
        best_k <- k_try
      }
    }
  }

  # Final prediction on test data
  k_final <- best_k
  assertthat::assert_that(k_final <= nrow(X_all), msg = "k cannot be greater than the number of training samples")

  preds <- numeric(nrow(X_test))
  probs <- numeric(nrow(X_test))

  for (i in 1:nrow(X_test)) {
    dists <- sqrt(rowSums((t(t(X_all) - X_test[i, ]))^2))
    neighbor_idx <- order(dists)[1:k_final]
    prob <- mean(y_all[neighbor_idx])
    preds[i] <- ifelse(prob >= 0.5, 1, 0)
    probs[i] <- prob
  }

  if (tune == TRUE){
    cat(sprintf("Best k-value = %.3f", k_final))
  }

  return(list(
    prediction = as.numeric(preds),
    probability = as.numeric(probs),
    best_k = k_final
  ))
}
