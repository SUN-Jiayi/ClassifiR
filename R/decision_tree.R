#' Calculate Gini Impurity
#'
#' @description
#' Computes the Gini impurity of a binary vector. Useful for decision tree splitting criteria.
#'
#' @param y Binary target vector (0/1).
#' @return Numeric Gini impurity value.
#'
#' @examples
#' y = c(0, 1, 1, 0, 1)
#' calculate_gini(y)
#'
#' @export
calculate_gini = function(y) {
  assertthat::assert_that(is.numeric(y),
                          msg = "Input y must be a numeric vector.")
  assertthat::assert_that(all(y %in% c(0, 1)),
                          msg = "Input y must contain only 0 and 1.")

  if (length(y) == 0) return(0)
  p = mean(y == 1)
  gini = 1 - p^2 - (1 - p)^2
  return(gini)
}


#' Calculate Entropy Impurity
#'
#' @description
#' Computes the entropy impurity of a binary vector. Useful for decision tree splitting criteria.
#'
#' @param y Binary target vector (0/1).
#' @return Numeric entropy impurity value (in bits).
#'
#' @examples
#' y = c(0, 1, 1, 0, 1)
#' calculate_entropy(y)
#'
#' @export
calculate_entropy = function(y) {
  assertthat::assert_that(is.numeric(y),
                          msg = "Input y must be a numeric vector.")
  assertthat::assert_that(all(y %in% c(0, 1)),
                          msg = "Input y must contain only 0 and 1.")

  if (length(y) == 0) return(0)
  p = mean(y == 1)
  if (p == 0 || p == 1) return(0)  # log(0)
  entropy = - p * log2(p) - (1 - p) * log2(1 - p)
  return(entropy)
}


#' Calculate Impurity Based on Specified Criterion
#'
#' @description
#' Computes impurity (Gini or Entropy) of a binary target vector based on the selected criterion.
#'
#' @param y Binary target vector (0/1).
#' @param criterion Impurity criterion to use: "gini" or "entropy".
#' @return Numeric impurity value.
#'
#' @examples
#' y = c(0, 1, 1, 0, 1)
#' calculate_impurity(y, "gini")
#' calculate_impurity(y, "entropy")
#'
#' @export
calculate_impurity = function(y, criterion = "gini") {

  assertthat::assert_that(is.numeric(y),
                          msg = "Target y must be numeric.")

  assertthat::assert_that(all(y %in% c(0, 1)),
                          msg = "Input y must contain only 0 and 1.")

  assertthat::assert_that(length(y) > 0,
                          msg = "Target y must not be empty.")

  if (criterion == "gini") {
    return(calculate_gini(y))
  } else if (criterion == "entropy") {
    return(calculate_entropy(y))
  } else {
    stop("Invalid criterion. Choose 'gini' or 'entropy'.")
  }
}


#' Find the Best Split Point for a Single Feature
#'
#' @description
#' Identifies the optimal threshold to split a numeric feature by minimizing impurity (Gini or Entropy).
#'
#' @param x A numeric vector representing a single feature.
#' @param y A binary target vector (0/1).
#' @param criterion Impurity criterion to use, either "gini" or "entropy".
#' @return A list containing:
#' \describe{
#'   \item{split}{The best split point (numeric).}
#'   \item{impurity}{The weighted impurity for this split.}
#' }
#'
#' @examples
#' x = c(1, 2, 3, 4, 5)
#' y = c(0, 0, 1, 1, 1)
#' find_best_split_for_feature(x, y, criterion = "gini")
#'
#' @export
find_best_split_for_feature = function(x, y, criterion = "gini") {

  assertthat::assert_that(is.numeric(x),
                          msg="Feature x should be numeric.")

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == length(x),
    msg = "Target y must be a numeric vector of 0s and 1s with length equal to the length of the feature x."
  )

  split_candidates = sort(unique(x))
  if (length(split_candidates) <= 1) return(NULL)

  split_points = (split_candidates[-1] + split_candidates[-length(split_candidates)]) / 2

  best_impurity = Inf
  best_split = NULL

  for (s in split_points) {
    left_idx = which(x <= s)
    right_idx = which(x > s)

    if (length(left_idx) == 0 || length(right_idx) == 0) next

    left_y = y[left_idx]
    right_y = y[right_idx]

    impurity_left = calculate_impurity(left_y, criterion)
    impurity_right = calculate_impurity(right_y, criterion)

    weighted_impurity = (length(left_y) * impurity_left + length(right_y) * impurity_right) / (length(left_y) + length(right_y))

    if (weighted_impurity < best_impurity) {
      best_impurity = weighted_impurity
      best_split = s
    }
  }

  return(list(split = best_split, impurity = best_impurity))
}


#' Find the Best Feature and Split Point among All Features
#'
#' @description
#' Searches across all numeric features in a data frame to find the optimal feature and split point
#' that minimizes impurity (Gini or Entropy).
#'
#' @param X A data.frame with numeric features.
#' @param y A binary target vector (0/1).
#' @param criterion Impurity criterion to use, either "gini" or "entropy".
#' @return A list containing:
#' \describe{
#'   \item{feature}{The best feature name (character).}
#'   \item{split_value}{The best threshold value for splitting (numeric).}
#' }
#'
#' @examples
#' X = data.frame(
#'   x1 = c(1, 2, 3, 4, 5),
#'   x2 = c(5, 4, 3, 2, 1)
#' )
#' y = c(0, 0, 1, 1, 1)
#' find_best_split(X, y, criterion = "gini")
#'
#' @export
find_best_split = function(X, y, criterion = "gini") {
  assertthat::assert_that(is.data.frame(X),
                          all(sapply(X, is.numeric)),
                          msg = "X must be a data frame and all columns in X must be numeric.")

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == nrow(X),
    msg = "Target y must be a numeric vector of 0s and 1s with length equal to number of rows in X."
  )

  best_feature = NULL
  best_split = NULL
  best_impurity = Inf

  for (feature in names(X)) {
    if (!is.numeric(X[[feature]])) next

    result = find_best_split_for_feature(X[[feature]], y, criterion)
    if (!is.null(result) && result$impurity < best_impurity) {
      best_feature = feature
      best_split = result$split
      best_impurity = result$impurity
    }
  }

  if (is.null(best_feature)) {
    return(NULL)
  }

  return(list(
    feature = best_feature,
    split_value = best_split
  ))
}


#' Build a Decision Tree Classifier
#'
#' @description
#' This function recursively builds a binary decision tree for binary classification problems.
#' It uses either Gini impurity or Entropy as the splitting criterion and supports maximum depth and minimum sample split control.
#'
#' @param X A data.frame containing feature columns (must be numeric).
#' @param y A binary target vector (0/1 or TRUE/FALSE).
#' @param depth Current depth of the tree (internal use, default = 0).
#' @param max_depth Maximum depth allowed for the tree (default = 5).
#' @param min_samples_split Minimum number of samples required to split a node (default = 2).
#' @param criterion The impurity criterion to use: "gini" or "entropy" (default = "gini").
#'
#' @return A nested list representing the decision tree structure. Each node contains:
#' \describe{
#'   \item{type}{Either "node" for internal nodes or "leaf" for terminal nodes.}
#'   \item{feature}{(for node) The feature used for splitting.}
#'   \item{value}{(for node) The threshold value used for splitting.}
#'   \item{left}{(for node) The left subtree.}
#'   \item{right}{(for node) The right subtree.}
#'   \item{class}{(for leaf) Predicted class (0 or 1).}
#'   \item{prob}{(for leaf) Predicted probability of class 1.}
#' }
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Train a decision tree
#' build_tree(X, y, max_depth = 2, min_samples_split = 2, criterion = "gini")
#'
#' @export
build_tree = function(X, y, depth = 0, max_depth = 5, min_samples_split = 2, criterion = "gini") {
  assertthat::assert_that(is.data.frame(X),
                          all(sapply(X, is.numeric)),
                          msg = "X must be a data frame and all columns in X must be numeric.")

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == nrow(X),
    msg = "Target y must be a numeric vector of 0s and 1s with length equal to number of rows in X."
  )

  assertthat::assert_that(
    is.numeric(depth), length(depth) == 1, depth >= 0,
    msg = "depth must be a single non-negative numeric value."
  )

  assertthat::assert_that(
    is.numeric(max_depth), length(max_depth) == 1, max_depth >= 1,
    msg = "max_depth must be a single numeric value greater than or equal to 1."
  )

  assertthat::assert_that(
    is.numeric(min_samples_split), length(min_samples_split) == 1, min_samples_split >= 1,
    msg = "min_samples_split must be a single numeric value greater than or equal to 1."
  )

  assertthat::assert_that(
    is.character(criterion), length(criterion) == 1, criterion %in% c("gini", "entropy"),
    msg = "criterion must be either 'gini' or 'entropy'."
  )

  # the number of samples is too small
  if (nrow(X) < min_samples_split) {
    return(list(type = "leaf", class = as.integer(mean(y) >= 0.5), prob = mean(y)))
  }

  # leaf is pure
  if (length(unique(y)) == 1) {
    return(list(type = "leaf", class = unique(y), prob = unique(y)))
  }

  # reach the maximum depth
  if (depth >= max_depth) {
    return(list(type = "leaf", class = as.integer(mean(y) >= 0.5), prob = mean(y)))
  }

  split = find_best_split(X, y, criterion)

  # cannot find a valid split: all the features cannot be split
  if (is.null(split)) {
    return(list(type = "leaf", class = as.integer(mean(y) >= 0.5), prob = mean(y)))
  }

  # split the data
  feature = split$feature
  value = split$split_value
  left_idx = which(X[[feature]] <= value)
  right_idx = which(X[[feature]] > value)

  left_subtree = build_tree(X[left_idx, , drop = FALSE], y[left_idx], depth + 1, max_depth, min_samples_split, criterion)
  right_subtree = build_tree(X[right_idx, , drop = FALSE], y[right_idx], depth + 1, max_depth, min_samples_split, criterion)

  # return the current node
  return(list(
    type = "node",
    feature = feature,
    value = value,
    left = left_subtree,
    right = right_subtree
  ))
}



#' Predict One Sample Using the Decision Tree
#'
#' @description
#' Predicts the class label (0 or 1) for a single observation using a trained decision tree model.
#' The input sample must be provided as a named vector or list, with names matching the feature names used in training.
#'
#' The function recursively traverses the tree until it reaches a leaf node and returns the class prediction stored there.
#'
#' @param tree A trained decision tree (nested list structure returned by \code{build_tree}).
#' @param x A named vector or list representing one sample (must contain all required feature names).
#'
#' @return Predicted class label: 0 or 1.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Train a decision tree
#' tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
#'
#' # Predict a new sample
#' sample = as.list(X[1, ])
#' pred = predict_tree(tree, sample)
#' print(pred)
#'
#' @export
predict_tree = function(tree, x) {
  assertthat::assert_that(
    is.list(tree),
    !is.null(tree$type),
    tree$type %in% c("leaf", "node"),
    msg = "tree must be a list with type = 'leaf' or 'node'"
  )

  if (is.list(x)) {
    stopifnot(all(sapply(x, is.numeric)))
  } else if (is.numeric(x)) {
    x = as.list(x)
  } else {
    stop("x must be a named numeric vector or list.")
  }

  if (tree$type == "leaf") {
    return(tree$class)
  }

  feature = tree$feature
  value = tree$value

  if (x[[feature]] <= value) {
    return(predict_tree(tree$left, x))
  } else {
    return(predict_tree(tree$right, x))
  }
}


#' Predict Multiple Samples Using the Decision Tree
#'
#'@description
#' Predicts binary class labels for a batch of samples using a trained decision tree.
#' Each row in the input data is treated as a separate observation.
#'
#' @param tree A trained decision tree.
#' @param X Feature data.frame.
#' @return A vector of predicted classes.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Train a decision tree
#' tree = build_tree(X, y, max_depth = 2, min_samples_split = 2, criterion = "gini")
#'
#' # Predict labels for the data
#' preds = predict_tree_batch(tree, X)
#' print(preds)
#'
#' @export
predict_tree_batch = function(tree, X) {
  assertthat::assert_that(
    is.list(tree),
    !is.null(tree$type),
    tree$type %in% c("leaf", "node"),
    msg = "tree must be a list with type = 'leaf' or 'node'"
  )

  assertthat::assert_that(is.data.frame(X),
                          all(sapply(X, is.numeric)),
                          msg = "X must be a data frame and all columns in X must be numeric.")

  preds = apply(X, 1, function(row) {
    predict_tree(tree, as.list(row))
  })

  return(as.integer(preds))
}


#' Tune Decision Tree Hyperparameters
#'
#'@description
#' Performs a grid search over maximum depth and minimum split size to find the best decision tree configuration,
#' based on validation accuracy.
#'
#' @param X Feature data.frame.
#' @param y Target vector.
#' @param max_depth_values Vector of candidate max_depth values.
#' @param min_samples_split_values Vector of candidate min_samples_split values.
#' @param criterion Splitting criterion ("gini" or "entropy").
#' @param split_ratio Proportion of data used for training (default 0.8).
#' @param seed Random seed for reproducibility.
#'
#' @return A list with best parameters, best model, and best validation accuracy.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Tune a decision tree model
#' tuning_result = tune_decision_tree(
#'   X, y,
#'   max_depth_values = c(2, 3),
#'   min_samples_split_values = c(2, 3),
#'   criterion = "gini",
#'   split_ratio = 0.8,
#'   seed = 123
#' )
#'
#' # Print best parameters found
#' print(tuning_result$best_params)
#' print(paste("Best Validation Accuracy:", round(tuning_result$best_score, 4)))
#'
#' @export
tune_decision_tree = function(X, y,
                              max_depth_values = c(3, 5, 7, 9, 11),
                              min_samples_split_values = c(20, 30, 50),
                              criterion = "gini",
                              split_ratio = 0.8,
                              seed = 123) {

  assertthat::assert_that(
    is.data.frame(X),
    all(sapply(X, is.numeric)),
    msg = "X must be a data.frame where all columns are numeric."
  )

  assertthat::assert_that(
    is.numeric(y),
    all(y %in% c(0, 1)),
    length(y) == nrow(X),
    msg = "Target y must be a numeric vector of 0s and 1s with length equal to number of rows in X."
  )

  assertthat::assert_that(
    is.numeric(max_depth_values),
    all(max_depth_values > 0),
    all(max_depth_values == as.integer(max_depth_values)),
    msg = "max_depth_values must be a vector of positive integers."
  )

  assertthat::assert_that(
    is.numeric(min_samples_split_values),
    all(min_samples_split_values >= 2),
    all(min_samples_split_values == as.integer(min_samples_split_values)),
    msg = "min_samples_split_values must be a vector of integers >= 2."
  )

  assertthat::assert_that(
    criterion %in% c("gini", "entropy"),
    msg = "criterion must be either 'gini' or 'entropy'."
  )

  assertthat::assert_that(
    is.numeric(split_ratio),
    length(split_ratio) == 1,
    split_ratio > 0,
    split_ratio < 1,
    msg = "split_ratio must be a number strictly between 0 and 1."
  )

  assertthat::assert_that(
    is.numeric(seed),
    length(seed) == 1,
    seed >= 1,
    msg = "seed must be a non-negative numeric scalar."
  )


  set.seed(seed)
  n = nrow(X)
  train_idx = sample(1:n, size = split_ratio * n)
  X_train = X[train_idx, , drop = FALSE]
  y_train = y[train_idx]
  X_val = X[-train_idx, , drop = FALSE]
  y_val = y[-train_idx]

  best_score = -Inf
  best_model = NULL
  best_params = NULL

  for (depth in max_depth_values) {
    for (minsplit in min_samples_split_values) {
      model = build_tree(X_train, y_train, max_depth = depth, min_samples_split = minsplit, criterion = criterion)
      preds = predict_tree_batch(model, X_val)
      acc = mean(preds == y_val)

      if (acc > best_score) {
        best_score = acc
        best_model = model
        best_params = list(max_depth = depth, min_samples_split = minsplit)
      }
    }
  }

  return(list(
    best_params = best_params,
    best_model = best_model,
    best_score = best_score
  ))
}


#' Predict Probability for a Single Observation Using Decision Tree
#'
#'@description
#' Traverses a trained decision tree to predict the probability of the positive class (y = 1) for a single observation.
#' Returns the probability value stored in the corresponding leaf node.
#'
#' @param tree A trained decision tree.
#' @param x A named list or vector representing one sample.
#' @return Predicted probability of y = 1
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Train a decision tree
#' tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
#'
#' # Predict probability for a single sample
#' new_sample = as.list(X[1, ])
#' prob = predict_tree_prob(tree, new_sample)
#' print(prob)
#'
#' @export
predict_tree_prob = function(tree, x) {
  assertthat::assert_that(
    is.list(tree),
    !is.null(tree$type),
    tree$type %in% c("leaf", "node"),
    msg = "tree must be a list with type = 'leaf' or 'node'"
  )

  if (is.list(x)) {
    stopifnot(all(sapply(x, is.numeric)))
  } else if (is.numeric(x)) {
    x = as.list(x)
  } else {
    stop("x must be a named numeric vector or list.")
  }


  if (tree$type == "leaf") {
    return(tree$prob)
  }

  feature = tree$feature
  value = tree$value

  if (x[[feature]] <= value) {
    return(predict_tree_prob(tree$left, x))
  } else {
    return(predict_tree_prob(tree$right, x))
  }
}



#' Predict Probabilities for Multiple Observations Using Decision Tree
#'
#'@description
#' Predicts the probability of the positive class (y = 1) for each sample in a dataset,
#' by traversing a trained decision tree. Applies \code{predict_tree_prob} to each observation in the input data.
#'
#' @param tree A trained decision tree.
#' @param X A feature data.frame.
#' @return A numeric vector of predicted probabilities.
#'
#' @examples
#' # Create toy data
#' X = data.frame(
#'   x1 = c(2, 5, 4, 3, 7, 2),
#'   x2 = c(1, 2, 1, 1, 3, 2)
#' )
#' y = c(0, 0, 1, 1, 1, 0)
#'
#' # Train a decision tree
#' tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
#'
#' # Predict probabilities for all samples
#' probs = predict_tree_prob_batch(tree, X)
#' print(probs)
#'
#' @export
predict_tree_prob_batch = function(tree, X) {
  assertthat::assert_that(
    is.list(tree),
    !is.null(tree$type),
    tree$type %in% c("leaf", "node"),
    msg = "tree must be a list with type = 'leaf' or 'node'"
  )

  assertthat::assert_that(is.data.frame(X),
                          all(sapply(X, is.numeric)),
                          msg = "X must be a data frame and all columns in X must be numeric.")

  probs = apply(X, 1, function(row) {
    predict_tree_prob(tree, as.list(row))
  })

  return(as.numeric(probs))
}
