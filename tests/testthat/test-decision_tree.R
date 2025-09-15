# Check if an error message is thrown if the input is invalid
test_that("calculate_gini input validation", {
  expect_error(calculate_gini(NULL))
  expect_error(calculate_gini(c(0, 1, 2)))
})

test_that("calculate_entropy input validation", {
  expect_error(calculate_entropy(NULL))
  expect_error(calculate_entropy(c(0, 1, 3)))
})

test_that("calculate_impurity input validation", {
  expect_error(calculate_impurity(NULL))
  expect_error(calculate_impurity(c(0, 2)))
  expect_error(calculate_impurity(c(0, 1), criterion = "invalid"))
})

test_that("find_best_split_for_feature input validation", {
  expect_error(find_best_split_for_feature("a", c(0, 1)))
  expect_error(find_best_split_for_feature(c(1, 2), c(0, 2)))
  expect_error(find_best_split_for_feature(c(1, 2), c(0, 1, 1)))
})

test_that("find_best_split input validation", {
  X = data.frame(a = c(1, 2), b = c("x", "y"))
  expect_error(find_best_split(X, c(0, 1)))
  expect_error(find_best_split(data.frame(a = 1:3), c(0, 1)))
})

test_that("build_tree input validation", {
  expect_error(build_tree(1, c(0, 1)))
  expect_error(build_tree(data.frame(x = c(1, 2)), c(0, 1), depth = -1))
  expect_error(build_tree(data.frame(x = c(1, 2)), c(0, 1), max_depth = 0))
  expect_error(build_tree(data.frame(x = c(1, 2)), c(0, 1), min_samples_split = 0))
  expect_error(build_tree(data.frame(x = c(1, 2)), c(0, 1), criterion = "other"))
})

test_that("predict_tree input validation", {
  expect_error(predict_tree(NULL, c(a = 1)))
  expect_error(predict_tree(list(type = "invalid"), c(a = 1)))
  expect_error(predict_tree(list(type = "node", feature = "a", value = 1, left = NULL, right = NULL), x = list()))
})

test_that("predict_tree_batch input validation", {
  expect_error(predict_tree_batch(NULL, data.frame(a = 1)))
  expect_error(predict_tree_batch(list(type = "node"), 1))
})

test_that("tune_decision_tree input validation", {
  expect_error(tune_decision_tree(1, c(0, 1)))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1)))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1, 1), max_depth_values = c(0)))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1, 1), min_samples_split_values = c(1)))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1, 1), criterion = "other"))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1, 1), split_ratio = 1.5))
  expect_error(tune_decision_tree(data.frame(a = 1:3), c(0, 1, 1), seed = -1))
})

test_that("predict_tree_prob input validation", {
  expect_error(predict_tree_prob(NULL, c(a = 1)))
  expect_error(predict_tree_prob(list(type = "node"), "not_a_vector"))
})

test_that("predict_tree_prob_batch input validation", {
  expect_error(predict_tree_prob_batch(NULL, data.frame(a = 1)))
  expect_error(predict_tree_prob_batch(list(type = "node"), 1))
})

# Check whether outputs of the functions are valid

test_that("calculate_gini works correctly", {
  expect_equal(calculate_gini(c(1, 1, 0, 0)), 0.5)
  expect_equal(calculate_gini(c(1, 1, 1, 1)), 0)
})

test_that("calculate_entropy works correctly", {
  expect_equal(round(calculate_entropy(c(1, 0)), 3), 1.000)
  expect_equal(calculate_entropy(c(1, 1, 1, 1)), 0)
})

test_that("calculate_impurity handles criteria correctly", {
  expect_equal(calculate_impurity(c(0, 1, 1, 0), criterion = "gini"), 0.5)
  expect_equal(round(calculate_impurity(c(0, 1), criterion = "entropy"), 3), 1.000)
  expect_error(calculate_impurity(c(0, 1), criterion = "invalid"))
})

test_that("find_best_split_for_feature works correctly", {
  x = c(1, 2, 3, 4, 5, 6)
  y = c(0, 0, 1, 1, 1, 1)

  result = find_best_split_for_feature(x, y)
  expect_type(result, "list")
  expect_named(result, c("split", "impurity"))
  expect_true(is.numeric(result$split))
  expect_true(result$impurity >= 0)
})

test_that("find_best_split returns valid feature and value", {
  X = data.frame(a = c(1, 2, 3, 4, 5), b = c(5, 4, 3, 2, 1))
  y = c(0, 0, 1, 1, 1)

  result = find_best_split(X, y)
  expect_type(result, "list")
  expect_named(result, c("feature", "split_value"))
  expect_true(result$feature %in% names(X))
  expect_true(is.numeric(result$split_value))
})

test_that("find_best_split returns NULL when no valid splits are possible", {
  X = data.frame(a = rep(1, 5))
  y = c(0, 0, 1, 1, 1)

  result = find_best_split(X, y)
  expect_null(result)
})

test_that("build_tree returns correct structure", {
  X = data.frame(x1 = c(1, 2, 3, 4), x2 = c(2, 3, 4, 5))
  y = c(0, 0, 1, 1)
  tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
  expect_true(is.list(tree))
  expect_true(all(c("type", "feature", "value") %in% names(tree)))
})

test_that("predict_tree and predict_tree_batch give expected outputs", {
  X = data.frame(x1 = c(1, 2, 3, 4), x2 = c(2, 3, 4, 5))
  y = c(0, 0, 1, 1)
  tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
  preds = predict_tree_batch(tree, X)
  expect_equal(length(preds), 4)
  expect_true(all(preds %in% c(0, 1)))
})

test_that("predict_tree_prob and predict_tree_prob_batch work", {
  X = data.frame(x1 = c(1, 2, 3, 4), x2 = c(2, 3, 4, 5))
  y = c(0, 0, 1, 1)
  tree = build_tree(X, y, max_depth = 2, min_samples_split = 2)
  prob = predict_tree_prob(tree, as.list(X[1,]))
  expect_true(prob >= 0 && prob <= 1)

  probs = predict_tree_prob_batch(tree, X)
  expect_equal(length(probs), 4)
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("tune_decision_tree returns expected structure", {
  X = data.frame(x1 = c(1, 2, 3, 4, 5, 6), x2 = c(2, 3, 4, 5, 6, 7))
  y = c(0, 0, 1, 1, 0, 1)
  result = tune_decision_tree(X, y, max_depth_values = c(2, 3), min_samples_split_values = c(2))
  expect_true("best_params" %in% names(result))
  expect_true("best_model" %in% names(result))
  expect_true("best_score" %in% names(result))
})
