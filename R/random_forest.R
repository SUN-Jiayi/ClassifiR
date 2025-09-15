#' Train a Random Forest Model
#'
#'@description
#' Trains a random forest classifier on a binary outcome using the \code{randomForest} package.
#' Allows control of number of trees, number of features considered per split, and terminal node size.
#'
#' @param X Feature data.frame.
#' @param y Target vector.
#' @param ntree Number of trees.
#' @param mtry Number of features to consider at each split.
#' @param nodesize Minimum size of terminal nodes.
#' @return Trained random forest model.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(1, 2, 3, 4),
#'   x2 = c(5, 6, 7, 8)
#' )
#' y = c(0, 0, 1, 1)
#'
#' # Train a random forest model
#' model = train_random_forest(X, y, ntree = 50)
#'
#' @export
train_random_forest = function(X, y, ntree = 100, mtry = NULL, nodesize = 1) {
  assertthat::assert_that(
    is.data.frame(X),
    all(sapply(X, is.numeric)),
    msg = "X must be a data.frame with only numeric columns."
  )

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == nrow(X),
    msg = "y must be a numeric vector of 0s and 1s with the same number of rows as X."
  )

  assertthat::assert_that(
    is.numeric(ntree),
    length(ntree) == 1,
    ntree >= 1,
    ntree == as.integer(ntree),
    msg = "ntree must be a single positive integer."
  )

  if (!is.null(mtry)) {
    assertthat::assert_that(
      is.numeric(mtry),
      length(mtry) == 1,
      mtry >= 1,
      mtry <= ncol(X),
      mtry == as.integer(mtry),
      msg = "mtry must be a positive integer no greater than the number of features in X."
    )
  }

  assertthat::assert_that(
    is.numeric(nodesize),
    length(nodesize) == 1,
    nodesize >= 1,
    nodesize == as.integer(nodesize),
    msg = "nodesize must be a positive integer."
  )

  model = randomForest::randomForest(
    x = X,
    y = as.factor(y),
    ntree = ntree,
    mtry = ifelse(is.null(mtry), floor(sqrt(ncol(X))), mtry),
    nodesize = nodesize
  )

  return(model)
}


#' Predict with Random Forest Model
#'
#'@description
#' Predicts binary class labels (0 or 1) for a given set of samples using a trained random forest model.
#'
#' @param model Trained random forest model.
#' @param X Feature data.frame for prediction.
#' @return Predicted classes (0/1).
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(1, 2, 3, 4),
#'   x2 = c(5, 6, 7, 8)
#' )
#' y = c(0, 0, 1, 1)
#'
#' # Train a random forest model
#' model = train_random_forest(X, y)
#'
#' # Predict the result using the trained random forest
#' preds = predict_random_forest(model, X)
#' print(preds)
#'
#' @export
predict_random_forest = function(model, X) {
  assertthat::assert_that(
    inherits(model, "randomForest"),
    msg = "model must be a trained random forest model of class 'randomForest'."
  )

  assertthat::assert_that(
    is.data.frame(X),
    all(sapply(X, is.numeric)),
    msg = "X must be a data.frame with only numeric columns."
  )

  preds = predict(model, X)
  return(as.integer(as.character(preds)))
}



#' Tune Random Forest Hyperparameters
#'
#'@description
#' Performs grid search over candidate \code{mtry} and \code{nodesize} values to select the best-performing random forest
#' model based on validation accuracy. The training set is split into train/validation according to \code{split_ratio}.
#'
#' @param X Feature data.frame.
#' @param y Target vector.
#' @param mtry_values A vector of candidate mtry values.
#' @param nodesize_values A vector of candidate nodesize values.
#' @param ntree Number of trees (fixed).
#' @param split_ratio Proportion for train/validation split.
#' @param seed Random seed.
#' @return List with best model, best params, best validation accuracy.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(1, 2, 3, 4, 5),
#'   x2 = c(5, 6, 7, 8, 9)
#' )
#' y = c(0, 0, 1, 1, 1)
#'
#' # Tune a random forest model
#' result = tune_random_forest(
#'   X, y,
#'   mtry_values = c(1, 2),
#'   nodesize_values = c(1, 2),
#'   ntree = 50,
#'   split_ratio = 0.8,
#'   seed = 123
#' )
#'
#' # Print best parameters found
#' print(result$best_params)
#' print(result$best_score)
#'
#' @export
tune_random_forest = function(X, y,
                              mtry_values = NULL,
                              nodesize_values = c(1, 5, 10),
                              ntree = 100,
                              split_ratio = 0.8,
                              seed = 123) {

  assertthat::assert_that(
    is.data.frame(X),
    all(sapply(X, is.numeric)),
    msg = "X must be a data frame with all numeric columns."
  )

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == nrow(X),
    msg = "y must be a numeric vector containing only 0 and 1, and its length must equal the number of rows in X."
  )

  if (!is.null(mtry_values)) {
    assertthat::assert_that(
      is.numeric(mtry_values),
      all(mtry_values > 0),
      all(mtry_values == as.integer(mtry_values)),
      msg = "mtry_values must be a vector of positive integers or NULL."
    )
  }

  assertthat::assert_that(
    is.numeric(nodesize_values),
    all(nodesize_values >= 1),
    all(nodesize_values == as.integer(nodesize_values)),
    msg = "nodesize_values must be a vector of integers >= 1."
  )

  assertthat::assert_that(
    is.numeric(ntree),
    length(ntree) == 1,
    ntree >= 1,
    ntree == as.integer(ntree),
    msg = "ntree must be a single positive integer."
  )

  assertthat::assert_that(
    is.numeric(split_ratio),
    length(split_ratio) == 1,
    split_ratio > 0,
    split_ratio < 1,
    msg = "split_ratio must be a numeric value strictly between 0 and 1."
  )

  assertthat::assert_that(
    is.numeric(seed),
    length(seed) == 1,
    seed == as.integer(seed),
    seed >= 1,
    msg = "seed must be a non-negative single integer."
  )

  set.seed(seed)
  n = nrow(X)
  train_idx = sample(1:n, size = split_ratio * n)
  X_train = X[train_idx, , drop = FALSE]
  y_train = y[train_idx]
  X_val = X[-train_idx, , drop = FALSE]
  y_val = y[-train_idx]

  if (is.null(mtry_values)) {
    p = ncol(X)
    mtry_values = c(floor(sqrt(p)/2), floor(sqrt(p)), floor(sqrt(p)*2))
  }

  best_score = -Inf
  best_model = NULL
  best_params = NULL

  for (mtry in mtry_values) {
    for (nodesize in nodesize_values) {
      model = randomForest::randomForest(
        x = X_train,
        y = as.factor(y_train),
        ntree = ntree,
        mtry = mtry,
        nodesize = nodesize
      )

      preds = predict(model, X_val)
      preds = as.integer(as.character(preds))
      acc = mean(preds == y_val)

      if (acc > best_score) {
        best_score = acc
        best_model = model
        best_params = list(mtry = mtry, nodesize = nodesize)
      }
    }
  }

  return(list(
    best_model = best_model,
    best_params = best_params,
    best_score = best_score
  ))
}


#' Predict Probability with Random Forest Model
#'
#'@description
#' Returns the predicted probabilities of the positive class (y = 1) for each sample using a trained random forest model.
#'
#' @param model Trained random forest model.
#' @param X Feature data.frame for prediction.
#' @return Vector of predicted probabilities for class 1.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(1, 2, 3, 4),
#'   x2 = c(5, 6, 7, 8)
#' )
#' y = c(0, 0, 1, 1)
#'
#' # Train a random forest model
#' model = train_random_forest(X, y)
#'
#' # Predict probability
#' probs = predict_random_forest_prob(model, X)
#' print(probs)
#'
#' @export
predict_random_forest_prob = function(model, X) {
  assertthat::assert_that(
    inherits(model, "randomForest"),
    msg = "model must be a trained random forest model of class 'randomForest'."
  )

  assertthat::assert_that(
    is.data.frame(X),
    all(sapply(X, is.numeric)),
    msg = "X must be a data.frame with only numeric columns."
  )

  prob_matrix = predict(model, X, type = "prob")
  return(prob_matrix[, "1"])  # extract P(y = 1)
}
