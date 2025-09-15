test_that("SVM returns predictions, probabilities, and margins", {
  set.seed(42)

  # Create synthetic dataset (100 samples, 5 numeric features)
  X <- matrix(rnorm(500), ncol = 5)
  y <- sample(0:1, 100, replace = TRUE)
  data <- data.frame(X, y = y)

  # Split into train/test
  train_idx <- sample(1:100, 70)
  train <- data[train_idx, ]
  test <- data[-train_idx, ]

  model <- my_svm(train, test)

  # Type and value checks
  expect_type(model, "list")
  expect_type(model$prediction, "double")
  expect_type(model$probability, "double")
  expect_true(all(model$prediction %in% c(0, 1)))
  expect_true(mean(model$prediction == test[[ncol(test)]]) >= 0)
  expect_true(all(model$probability >= 0 & model$probability <= 1))

  expect_equal(length(model$prediction), nrow(test))
  expect_equal(length(model$probability), nrow(test))
  expect_equal(length(model$prediction_train), nrow(train))
  expect_equal(length(model$probability_train), nrow(train))

  # Function works
  pred_from_fun <- model$predict_fun(as.matrix(test[, -ncol(test)]))
  expect_equal(length(pred_from_fun$preds), nrow(test))
  expect_equal(length(pred_from_fun$probs), nrow(test))
  expect_true(all(pred_from_fun$preds %in% c(0, 1)))
})


test_that("SVM handles missing values", {
  set.seed(42)

  # Create synthetic dataset (100 samples, 5 numeric features)
  X <- matrix(rnorm(500), ncol = 5)
  y <- sample(0:1, 100, replace = TRUE)
  data <- data.frame(X, y = y)

  # Split into train/test
  train_idx <- sample(1:100, 70)
  train <- data[train_idx, ]
  test <- data[-train_idx, ]

  train[1, 1] <- NA

  expect_error(my_svm(train, test), "Training data has NAs")
  expect_error(my_svm(test, train), "Test data has NAs")
})


test_that("SVM rejects non-binary labels", {
  train_bad <- data.frame(matrix(rnorm(20), nrow = 10))
  train_bad$label <- 2:11
  test_bad <- train_bad

  expect_error(
    my_knn(train_bad, test_bad),
    "Training labels must be binary (0 or 1).",
    fixed = TRUE
  )
})

test_that("SVM throws error on empty training/test data", {
  train <- data.frame()
  test <- data.frame()

  expect_error(my_svm(train, test), "Training data is empty.")
})


test_that("SVM throws error on non-numeric features", {
  train <- data.frame(x1 = c("a", "b"), x2 = c("c", "d"), y = c(0, 1))
  test <- data.frame(x1 = c("e", "f"), x2 = c("g", "h"), y = c(1, 0))

  expect_error(my_svm(train, test), "non-numeric argument")
})


test_that("SVM works with test set containing only one class", {
  train <- data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rep(c(0, 1), 10))
  test <- data.frame(x1 = rnorm(5), x2 = rnorm(5), y = rep(1, 5))

  model <- my_svm(train, test)
  expect_equal(length(model$prediction), nrow(test))
  expect_true(all(model$prediction %in% c(0, 1)))
})


test_that("SVM predict_fun output structure is valid", {
  train <- data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rep(c(0, 1), 10))
  test <- data.frame(x1 = rnorm(10), x2 = rnorm(10), y = rep(c(0, 1), 5))

  model <- my_svm(train, test)
  output <- model$predict_fun(as.matrix(test[, -ncol(test)]))

  expect_equal(sort(names(output)), sort(c("preds", "probs", "margins")))
  expect_equal(length(output$probs), nrow(test))
  expect_equal(length(output$preds), nrow(test))
})
