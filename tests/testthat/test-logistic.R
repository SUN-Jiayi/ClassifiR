test_that("Logistic Regression returns predictions and probabilities", {
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

  model <- my_logistic_regression(train, test)

  # Original checks
  expect_type(model, "list")
  expect_type(model$prediction, "double")
  expect_type(model$probability, "double")
  expect_true(all(model$prediction %in% c(0, 1)))
  expect_true(mean(model$prediction == test[[ncol(test)]]) >= 0)
  expect_true(all(model$probability >= 0 & model$probability <= 1))

  # Additional checks
  expect_equal(length(model$prediction), nrow(test))
  expect_equal(length(model$probability), nrow(test))
  expect_equal(length(model$prediction_train), nrow(train))
  expect_equal(length(model$probability_train), nrow(train))
  expect_equal(length(model$prediction), length(model$probability))

  # Predict function works
  pred_from_fun <- model$predict_fun(as.matrix(test[, -ncol(test)]))
  expect_equal(length(pred_from_fun$preds), nrow(test))
  expect_equal(length(pred_from_fun$probs), nrow(test))
  expect_true(all(pred_from_fun$preds %in% c(0, 1)))
})


test_that("Logistic Regression handles missing values in training or test data", {
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

  # Inject NA into train and test
  train_data_with_NA <- train
  test_data_with_NA <- test
  train_data_with_NA[1, 1] <- NA
  test_data_with_NA[1, 1] <- NA

  expect_error(
    my_logistic_regression(train_data_with_NA, test),
    "Training data contains missing values"
  )

  expect_error(
    my_logistic_regression(train, test_data_with_NA),
    "Test data contains missing values"
  )
})


test_that("Logistic Regression rejects non-binary labels", {
  train_bad <- data.frame(matrix(rnorm(20), nrow = 10))
  train_bad$label <- 2:11  # Not 0/1
  test_bad <- train_bad

  expect_error(
    my_logistic_regression(train_bad, test_bad),
    "Target variable must be binary"
  )
})

test_that("Logistic Regression returns correct structure", {
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

  model <- my_logistic_regression(train, test)

  expect_named(model, c("theta", "predict_fun", "prediction_train",
                        "probability_train", "prediction", "probability",
                        "best_learning_rate", "best_n_iter"))
  expect_type(model$prediction, "double")
  expect_type(model$probability, "double")
  expect_true(all(model$prediction %in% c(0, 1)))
  expect_equal(length(model$prediction), nrow(test))
})


test_that("Logistic Regression throws error on empty training data", {
  test <- data.frame(x1 = rnorm(5), x2 = rnorm(5), y = c(0, 1, 0, 1, 0))
  train <- data.frame()

  expect_error(my_logistic_regression(train, test), "Training data must not be empty")
})


test_that("Logistic Regression throws error on non-numeric features", {
  train <- data.frame(x1 = c("a", "b"), x2 = c("c", "d"), y = c(0, 1))
  test <- data.frame(x1 = c("e", "f"), x2 = c("g", "h"), y = c(1, 0))

  expect_error(my_logistic_regression(train, test))
})


test_that("Logistic Regression predict_fun returns correct structure", {
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

  model <- my_logistic_regression(train, test)

  output <- model$predict_fun(as.matrix(test[, -ncol(test)]))
  expect_named(output, c("probs", "preds"))
  expect_equal(length(output$probs), nrow(test))
  expect_equal(length(output$preds), nrow(test))
})


test_that("Logistic Regression handles test data with one class", {
  train <- data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rep(c(0, 1), 10))
  test <- data.frame(x1 = rnorm(10), x2 = rnorm(10), y = rep(1, 10))

  model <- my_logistic_regression(train, test)

  expect_true(all(model$prediction %in% c(0, 1)))
  expect_equal(length(model$prediction), nrow(test))
})
