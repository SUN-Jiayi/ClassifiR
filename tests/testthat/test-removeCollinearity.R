test_that("process_collinearity_ basic VIF works without removal", {
  df <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100)
  )
  df$x4 <- df$x1 * 0.99 + rnorm(100, 0, 0.001)

  result <- process_collinearity_(df, remove = FALSE)

  expect_type(result$vif, "double")
  expect_named(result$vif)
  expect_equal(length(result$vif), 4)
  expect_equal(length(result$removed), 0)
})

test_that("process_collinearity_ removes high VIF columns", {
  df <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100)
  )
  df$x4 <- df$x1 * 0.95 + rnorm(100, 0, 0.01)

  result <- process_collinearity_(df, threshold = 5, remove = TRUE)

  expect_lt(max(result$vif, na.rm = TRUE), 5)
  expect_true("x4" %in% result$removed || "x1" %in% result$removed)
})


test_that("non-numeric columns throw an error", {
  df <- data.frame(
    x1 = rnorm(100),
    x2 = letters[1:100]
  )
  expect_error(process_collinearity_(df), "All columns should be numeric")
})

test_that("missing values throw an error", {
  df <- data.frame(
    x1 = c(1, 2, NA),
    x2 = c(4, 5, 6)
  )
  expect_error(process_collinearity_(df), "data frame cannot contain missing values")
})

test_that("returns Inf for perfect multicollinearity", {
  df <- data.frame(
    x1 = rnorm(100)
  )
  df$x2 <- df$x1
  result <- process_collinearity_(df, remove = FALSE)
  expect_true(any(is.infinite(result$vif)))
})

test_that("data with only one column returns error", {
  df <- data.frame(x1 = rnorm(100))
  expect_error(process_collinearity_(df), "At least two numeric columns")
})

test_that("non-numeric threshold raises error", {
  df <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  expect_error(process_collinearity_(df, threshold = "high"), "threshold must be a numeric value larger than 1")
})

test_that("threshold smaller than 1 raises error", {
  df <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  expect_error(process_collinearity_(df, threshold = 0.5), "threshold must be a numeric value larger than 1")
  expect_error(process_collinearity_(df, threshold = 1), "threshold must be a numeric value larger than 1")
})

test_that("non-logical 'remove' raises error", {
  df <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  expect_error(process_collinearity_(df, remove = "yes"), "remove should be either TRUE or FALSE")
  expect_error(process_collinearity_(df, remove = 1), "remove should be either TRUE or FALSE")
})

test_that("returns correct names and structure", {
  df <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
  res <- process_collinearity_(df, remove = FALSE)
  expect_named(res, c("data", "vif", "removed"))
  expect_s3_class(res$data, "data.frame")
  expect_type(res$vif, "double")
  expect_null(res$removed)
})

test_that("data with all independent variables gives low VIF", {
  df <- as.data.frame(matrix(rnorm(1000), ncol = 5))
  colnames(df) <- paste0("X", 1:5)
  res <- process_collinearity_(df, threshold = 5, remove = TRUE)
  expect_equal(length(res$removed), 0)
  expect_true(all(res$vif < 5))
})
