# Check if an error message is thrown if the input is invalid
test_that("train_random_forest errors on invalid inputs", {
  X = data.frame(x1 = 1:4, x2 = 5:8)
  y = c(0, 1, 0, 1)

  expect_error(train_random_forest("not a data frame", y))
  expect_error(train_random_forest(X, c("a", "b", "c", "d")))
  expect_error(train_random_forest(X, y[1:2]))  # mismatched length
  expect_error(train_random_forest(X, y, ntree = -10))
  expect_error(train_random_forest(X, y, mtry = 1000))
  expect_error(train_random_forest(X, y, nodesize = -1))
})

test_that("predict_random_forest errors on invalid inputs", {
  X = data.frame(x1 = 1:4, x2 = 5:8)
  y = c(0, 1, 0, 1)
  model = train_random_forest(X, y)

  expect_error(predict_random_forest("not a model", X))
  expect_error(predict_random_forest(model, "not a data frame"))
})

test_that("tune_random_forest errors on invalid inputs", {
  X = data.frame(x1 = 1:4, x2 = 5:8)
  y = c(0, 1, 0, 1)

  expect_error(tune_random_forest("not a data frame", y))
  expect_error(tune_random_forest(X, c("a", "b", "c", "d")))
  expect_error(tune_random_forest(X, y[1:2]))  # mismatched length
  expect_error(tune_random_forest(X, y, mtry_values = c(-1)))
  expect_error(tune_random_forest(X, y, nodesize_values = c(0)))
  expect_error(tune_random_forest(X, y, ntree = -10))
  expect_error(tune_random_forest(X, y, split_ratio = 1.5))
  expect_error(tune_random_forest(X, y, seed = "not a number"))
})

test_that("predict_random_forest_prob errors on invalid inputs", {
  X = data.frame(x1 = 1:4, x2 = 5:8)
  y = c(0, 1, 0, 1)
  model = train_random_forest(X, y)

  expect_error(predict_random_forest_prob("not a model", X))
  expect_error(predict_random_forest_prob(model, "not a data frame"))
})

# Check whether outputs of the functions are valid

test_that("train_random_forest returns a model and predictions work", {
  X = data.frame(
    x1 = c(1, 2, 3, 4),
    x2 = c(5, 6, 7, 8)
  )
  y = c(0, 0, 1, 1)
  model = train_random_forest(X, y, ntree = 10)
  expect_s3_class(model, "randomForest")
  preds = predict_random_forest(model, X)
  expect_type(preds, "integer")
  expect_length(preds, nrow(X))
  expect_true(all(preds %in% c(0, 1)))
})

test_that("predict_random_forest returns integer 0/1 vector", {
  X = data.frame(x1 = c(1, 2, 3, 4), x2 = c(5, 6, 7, 8))
  y = c(0, 0, 1, 1)
  model = train_random_forest(X, y, ntree = 10)

  preds = predict_random_forest(model, X)

  expect_type(preds, "integer")
  expect_true(all(preds %in% c(0, 1)))
  expect_length(preds, nrow(X))
})

test_that("predict_random_forest_prob returns probabilities between 0 and 1", {
  X = data.frame(
    x1 = c(1, 2, 3, 4),
    x2 = c(5, 6, 7, 8)
  )
  y = c(0, 0, 1, 1)
  model = train_random_forest(X, y, ntree = 10)
  probs = predict_random_forest_prob(model, X)
  expect_type(probs, "double")
  expect_length(probs, nrow(X))
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("tune_random_forest returns best parameters and a model", {
  X = data.frame(
    x1 = c(1, 2, 3, 4, 5),
    x2 = c(5, 6, 7, 8, 9)
  )
  y = c(0, 0, 1, 1, 1)
  result = tune_random_forest(
    X, y,
    mtry_values = c(1, 2),
    nodesize_values = c(1, 2),
    ntree = 10,
    split_ratio = 0.8,
    seed = 123
  )
  expect_true("best_model" %in% names(result))
  expect_true("best_params" %in% names(result))
  expect_true("best_score" %in% names(result))
  expect_s3_class(result$best_model, "randomForest")
})
