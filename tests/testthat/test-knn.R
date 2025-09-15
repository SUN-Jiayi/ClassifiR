test_that("KNN returns predictions and probabilities", {
  train <- data.frame(
    X1 = c(0, 1, 0, 1),
    X2 = c(0, 0, 1, 1),
    Label = c(0, 0, 1, 1)
  )

  test <- data.frame(
    X1 = c(0, 1),
    X2 = c(1, 0),
    Label = c(1, 0)
  )

  model <- my_knn(train, test)

  expect_type(model, "list")
  expect_type(model$prediction, "double")
  expect_type(model$probability, "double")
  expect_true(all(model$prediction %in% c(0, 1)))
  expect_true(mean(model$prediction == test[[ncol(test)]]) >= 0)
  expect_true(all(model$probability >= 0 & model$probability <= 1))

  expect_equal(length(model$prediction), nrow(test))
  expect_equal(length(model$probability), nrow(test))
})


test_that("KNN handles missing values", {
  train <- data.frame(
    X1 = c(0, 1, 0, 1),
    X2 = c(0, 0, 1, 1),
    Label = c(0, 0, 1, 1)
  )

  test <- data.frame(
    X1 = c(0, 1),
    X2 = c(1, 0),
    Label = c(1, 0)
  )

  train[1, 1] <- NA

  expect_error(my_knn(train, test), "Training data contains missing values")
  expect_error(my_knn(test, train), "Test data contains missing values.")
})


test_that("KNN rejects non-binary labels", {
  train_bad <- data.frame(matrix(rnorm(20), nrow = 10))
  train_bad$label <- 2:11
  test_bad <- train_bad

  expect_error(
    my_knn(train_bad, test_bad),
    "Training labels must be binary (0 or 1).",
    fixed = TRUE
  )
})

test_that("KNN throws error if k > number of training samples", {
  train <- data.frame(x1 = rnorm(3), x2 = rnorm(3), y = c(0, 1, 0))
  test <- data.frame(x1 = rnorm(2), x2 = rnorm(2), y = c(0, 1))

  expect_error(my_knn(train, test, k = 10), "k cannot be greater than the number of training samples")
})


test_that("KNN throws error on empty training/test data", {
  train <- data.frame()
  test <- data.frame()

  expect_error(my_knn(train, test), "Training data must not be empty")
})


test_that("KNN throws error on non-numeric features", {
  train <- data.frame(x1 = c("a", "b"), x2 = c("c", "d"), y = c(0, 1))
  test <- data.frame(x1 = c("e", "f"), x2 = c("g", "h"), y = c(1, 0))

  expect_error(my_knn(train, test), "All training features must be numeric.", fixed = TRUE)
})
