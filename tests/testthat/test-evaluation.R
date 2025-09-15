# Check if an error message is thrown if the input is invalid
test_that("evaluate_binary_classification handles invalid input", {
  expect_error(evaluate_binary_classification(c(0, 1), c(0, 1, 1)))
  expect_error(evaluate_binary_classification(c(0, 1), c(0, 2)))
  expect_error(evaluate_binary_classification(c(0, 1), c(0, 1), c(0.2, 1.2)))
})

test_that("evaluate_roc_auc handles invalid input", {
  expect_error(evaluate_roc_auc(c(0, 1, 2), c(0.1, 0.9, 0.3)))
  expect_error(evaluate_roc_auc(c(0, 1, 1), c(0.1, 0.9)))
})

test_that("plot_model_roc_comparison handles invalid input", {
  y = c(0, 1, 1, 0)
  prob_list = list(model1 = c(0.2, 0.8, 0.9, 0.1),
                   model2 = c(0.3, NA, 0.7, 0.4))
  expect_error(plot_model_roc_comparison(y, prob_list))
})

test_that("compare_models handles invalid input", {
  y = c(0, 1, 1, 0)
  preds = list(tree = c(0, 1, 1))
  probs = list(tree = c(0.2, 0.8, 0.9, 0.1), rf = c(0.2, 0.7, 0.8, 0.3))
  expect_error(compare_models(y, preds))
  expect_error(compare_models(y, list(tree = c(0, 1, 1, 2))))
  expect_error(compare_models(y, list(tree = c(0, 1, 1, 1)), probs))
})

# Check whether outputs of the functions are valid
test_that("evaluate_binary_classification returns valid structure", {
  y_true <- c(0, 1, 1, 0, 1, 0, 1, 0)
  y_pred <- c(0, 1, 1, 0, 0, 0, 1, 1)
  y_prob <- c(0.1, 0.9, 0.85, 0.2, 0.4, 0.3, 0.8, 0.7)

  result <- evaluate_binary_classification(y_true, y_pred, y_prob)

  expect_type(result, "list")
  expect_true(all(c("accuracy", "precision", "recall", "f1", "confusion_matrix") %in% names(result)))
  expect_type(result$log_loss, "double")
})

test_that("evaluate_roc_auc returns valid roc and auc", {
  y_true <- c(0, 1, 1, 0, 1, 0, 0, 1)
  y_prob <- c(0.2, 0.9, 0.8, 0.3, 0.7, 0.1, 0.4, 0.95)

  result <- evaluate_roc_auc(y_true, y_prob)

  expect_true("roc" %in% names(result))
  expect_true("auc" %in% names(result))
  expect_s3_class(result$roc, "roc")
  expect_type(result$auc, "double")
})

test_that("plot_model_roc_comparison returns a ggplot object", {
  y = c(0, 1, 1, 0, 1, 0, 1, 0)
  preds = list(
    DecisionTree = c(0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4),
    RandomForest = c(0.1, 0.9, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3)
  )

  p = plot_model_roc_comparison(y, preds)
  expect_s3_class(p, "gg")
})


test_that("compare_models returns correct structure", {
  y <- c(0, 1, 1, 0, 1, 0, 1, 0)
  preds <- list(
    DecisionTree = c(0, 1, 1, 0, 1, 0, 0, 0),
    RandomForest = c(0, 1, 1, 0, 1, 0, 1, 0)
  )
  probs <- list(
    DecisionTree = c(0.2, 0.9, 0.85, 0.1, 0.95, 0.3, 0.4, 0.5),
    RandomForest = c(0.1, 0.95, 0.87, 0.2, 0.97, 0.1, 0.88, 0.3)
  )

  result <- compare_models(y, preds, prob_list = probs)

  expect_type(result, "list")
  expect_true("summary_table" %in% names(result))
  expect_true("best_models" %in% names(result))
  expect_s3_class(result$summary_table, "data.frame")
  expect_type(result$best_models, "list")
})
