library(testthat)

test_that("adaboost_ works on simple binary data", {
  set.seed(123)
  data <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  data$label <- ifelse(data$x1 + data$x2 > 0, 1, 0)

  model <- adaboost_(data, n_iter = 10)
  expect_type(model, "list")
  expect_true(length(model$models) > 0)
  expect_equal(length(model$alphas), length(model$models))
})



test_that("adaboost_ throws an error when data is not a data frame", {
  expect_error(adaboost_(matrix(1:10, ncol = 2)), "'data' must be a data frame.")
})

test_that("adaboost_ throws an error when n_iter is not a positive integer", {
  data <- data.frame(x1 = rnorm(10), x2 = rnorm(10))
  data$label <- ifelse(data$x1 + data$x2 > 0, 1, 0)

  expect_error(adaboost_(data, n_iter = -1), "'n_iter' must be a positive integer.")
  expect_error(adaboost_(data, n_iter = "a"), "'n_iter' must be a positive integer.")
})

test_that("adaboost_ throws an error when data contains missing values", {
  data <- data.frame(x1 = rnorm(10), x2 = rnorm(10))
  data$label <- c(1, 0, 1, 1, 0, 1, NA, 0, 1, 0)  # Label with NA value

  expect_error(adaboost_(data), "Input 'data' contains missing values. Please remove or impute them.")
})

test_that("adaboost throws an error when label column has values other than 0 and 1", {
  data <- data.frame(x1 = rnorm(10), x2 = rnorm(10))
  data$label <- c(2, 0, 1, 1, 0, 0, 1, 2, 0, 1)  # Label with invalid values (2)

  expect_error(adaboost_(data), "Label column must contain only 0 and 1 values.")
})
test_that("adaboost throws an error when label column has values other than 0 and 1", {
  data <- data.frame(x1 = c(1, 2), x2 = c(1, 2), label = c(2, 0))
  expect_error(adaboost_(data), "Label column must contain only 0 and 1 values.")
})

test_that("adaboost_ throws error when data has only one column", {
  data <- data.frame(label = c(0, 1, 0, 1))  # only label column, no features
  expect_error(
    adaboost_(data),
    "Input 'data' must contain at least one feature and one label column."
  )
})
test_that("adaboost_ throws error when features are non-numeric", {
  data <- data.frame(
    color = c("red", "blue", "green"),
    label = c(1, 0, 1)
  )
  expect_error(
    adaboost_(data),
    "All feature columns must be numeric."
  )
})


test_that("predict_adaboost_ works with valid input", {
  model <- list(
    models = list(
      list(feature = 1, threshold = 1.5, polarity = 1),
      list(feature = 2, threshold = 3, polarity = -1)
    ),
    alphas = c(0.4, 0.6)
  )
  newdata <- data.frame(x1 = c(1, 2), x2 = c(3, 4))
  preds <- predict_adaboost_(model, newdata)
  expect_type(preds, "integer")
  expect_length(preds, nrow(newdata))
})

test_that("predict_adaboost_ returns binary predictions", {
  set.seed(123)
  data <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  data$label <- ifelse(data$x1 + data$x2 > 0, 1, 0)

  model <- adaboost_(data, n_iter = 5)
  pred <- predict_adaboost_(model, data[, -ncol(data)])
  expect_true(all(pred %in% c(0, 1)))
})

test_that("predict_adaboost_ handles model with no learners", {
  model <- list(models = list(), alphas = numeric())
  newdata <- data.frame(x = c(1, 2, 3))
  preds <- predict_adaboost_(model, newdata)
  expect_equal(preds, rep(0, 3))
})

test_that("predict_adaboost_ throws error on invalid model format", {
  bad_model <- list(a = 1)
  newdata <- data.frame(x = c(1, 2))
  expect_error(predict_adaboost_(bad_model, newdata), "Invalid model format")
})

test_that("predict_adaboost_ throws error when a stump is missing required fields", {
  bad_model <- list(
    models = list(
      list(feature = 1, threshold = 0.5),  # missing polarity
      list(threshold = 0.7, polarity = 1)  # missing feature
    ),
    alphas = c(0.3, 0.4)
  )
  newdata <- data.frame(x = c(1, 2))

  expect_error(
    predict_adaboost_(bad_model, newdata),
    "Each stump in model\\$models must be a list with 'feature', 'threshold', and 'polarity'."
  )
})

test_that("predict_adaboost_ throws error when alphas is invalid", {
  model <- list(
    models = list(list(feature = 1, threshold = 1, polarity = 1)),
    alphas = c(NA)
  )
  newdata <- data.frame(x1 = c(1, 2))
  expect_error(predict_adaboost_(model, newdata), "alphas.*contain no NA")
})

test_that("predict_adaboost_ throws error when newdata has fewer columns", {
  model <- list(
    models = list(list(feature = 2, threshold = 3, polarity = -1)),
    alphas = c(0.6)
  )
  newdata <- data.frame(x1 = c(1, 2))
  expect_error(predict_adaboost_(model, newdata), "New data has fewer columns")
})

test_that("predict_adaboost_ returns all-0 when no weak learners", {
  model <- list(models = list(), alphas = numeric(0))
  newdata <- data.frame(x1 = c(1, 2))
  expect_equal(predict_adaboost_(model, newdata), c(0L, 0L))
})


test_that("tune_adaboost_ works with valid input", {
  # Simulating training and validation data
  train_data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(1, 2, 3, 4), label = c(1, 0, 1, 0))
  val_data <- data.frame(x1 = c(2, 3), x2 = c(2, 3), label = c(0, 1))

  result <- tune_adaboost_(train_data, val_data, iter_options = c(10, 20, 30), verbose = TRUE)
  expect_type(result$model, "list")
  expect_true(result$n_iter %in% c(10, 20, 30))
  expect_equal(length(result$model$models), result$n_iter)
})

test_that("tune_adaboost_ throws error for non-numeric iter_options", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(1, 2), label = c(1, 0))
  val_data <- data.frame(x1 = c(1, 2), x2 = c(1, 2), label = c(0, 1))
  expect_error(tune_adaboost_(train_data, val_data, iter_options = c("10", 20, 30)),
               "'iter_options' must be a vector of positive integers.")
})

test_that("tune_adaboost_ throws error for non-binary target labels", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(3, 4), target = c(2, 1))
  val_data <- data.frame(x1 = c(1, 3), x2 = c(1, 2), label = c(0, 1))
  expect_error(tune_adaboost_(train_data, val_data),
               "Target variable must be binary")
})

test_that("tune_adaboost_ throws error if all models fail", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(1, 2), label = c(1, 1))
  val_data <- data.frame(x1 = c(1, 2), x2 = c(1, 2), label = c(1, 1))
  expect_error(tune_adaboost_(train_data, val_data, iter_options = c(20, 30)), "All models failed during tuning. Check data or parameters.")
})

test_that("tune_adaboost_ throws error if training feature columns are not numeric", {
  train_data <- data.frame(x1 = c("A", "B"), x2 = c(3, 4), target = c(0, 1))
  val_data <- data.frame(x1 = c(1, 2), x2 = c(5, 6), target = c(1, 0))
  expect_error(tune_adaboost_(train_data, val_data, iter_options = c(20, 30)),
               "All feature columns of training set must be numeric")
})

test_that("tune_adaboost_ throws error if validation feature columns are not numeric", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(3, 4), target = c(0, 1))
  val_data <- data.frame(x1 = c("1", 2), x2 = c(5, 6), target = c(1, 0))
  expect_error(tune_adaboost_(train_data, val_data, iter_options = c(20, 30)),
               "All feature columns of validation set must be numeric")
})

test_that("tune_adaboost_ throws error if validation labels are not binary", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(3, 4), target = c(0, 1))
  val_data <- data.frame(x1 = c(5, 6), x2 = c(7, 8), target = c(0, 2))
  expect_error(tune_adaboost_(train_data, val_data),
               "Validation target variable must be binary")
})

test_that("tune_adaboost_ throws error if training labels are not binary", {
  train_data <- data.frame(x1 = c(1, 2), x2 = c(3, 4), target = c(2, 1))
  val_data <- data.frame(x1 = c(5, 6), x2 = c(7, 8), target = c(0, 1))
  expect_error(tune_adaboost_(train_data, val_data),
               "Target variable must be binary")
})
