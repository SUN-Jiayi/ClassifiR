test_that("dataPreprocessing removes rows with NA", {
  df <- data.frame(A = c(1, 2, NA), B = c("a", "b", "c"))
  result <- dataPreprocessing(df, one_hot = FALSE, split_date = FALSE)
  expect_equal(nrow(result), 2)
})


test_that("dataPreprocessing splits Date column correctly", {
  df <- data.frame(Date = c("2001/01/01", "2002/02/02"), A = c(1, 2))
  result <- dataPreprocessing(df, one_hot = FALSE, split_date = TRUE)
  expect_true(all(c("Year", "Month", "Day") %in% colnames(result)))
  expect_equal(result$Year[1], 2001)
  expect_equal(result$Month[2], 2)
})

test_that("categorical columns are one-hot encoded", {
  df <- data.frame(A = c("Yes", "No", "Yes"), B = 1:3)
  result <- dataPreprocessing(df, one_hot = TRUE, split_date = FALSE)
  expect_true(any(grepl("A", names(result))))
  expect_equal(ncol(result), 2)  # one dummy + B
})

test_that("label column is moved to last (without one-hot encoding)", {
  df <- data.frame(A = c("Yes", "No", "Yes"), B = 1:3)
  result <- dataPreprocessing(df, one_hot = FALSE, split_date = FALSE, label_col = "A")
  expect_equal(names(result)[ncol(result)], "A")
})

test_that("label column is moved to last (with one-hot encoding)", {
  df <- data.frame(A = c("Yes", "No", "Yes"), B = 1:3)
  result <- dataPreprocessing(df, one_hot = TRUE, split_date = FALSE, label_col = "A")
  expect_equal(names(result)[ncol(result)], "A")
})

test_that("error when data is not a data.frame", {
  expect_error(dataPreprocessing(42), "'data' must be a data frame.")
})

test_that("error when label_col not in data", {
  df <- data.frame(A = 1:3, B = 4:6)
  expect_error(dataPreprocessing(df, label_col = "C"), "label_col does not exist in the dataset.")
})

test_that("error when label_col produces more than one dummy column", {
  df <- data.frame(A = c("yes", "no", "maybe"), B = 1:3)
  expect_error(dataPreprocessing(df, one_hot = TRUE, split_date = FALSE, label_col = "A"),
               "Label column after one-hot encoding must result in exactly one column.")
})

test_that("error if Date column contains invalid format", {
  df <- data.frame(Date = c("2020-01-01", "bad_date", "2021-03-01"), x = 1:3)
  expect_error(dataPreprocessing(df, split_date = TRUE), "could not be parsed")
})

test_that("error if all rows are removed due to NA", {
  df <- data.frame(A = c(NA, NA), B = c(1, 2))
  expect_error(dataPreprocessing(df), "All rows were removed after omitting NA values")
})

test_that("splitDataset_ splits data correctly", {
  df <- data.frame(id = 1:100, value = rnorm(100))

  result <- splitDataset_(df, test_ratio = 0.2, seed = 123)

  expect_true(is.list(result))
  expect_true(all(c("train", "test") %in% names(result)))
  expect_equal(nrow(result$train) + nrow(result$test), 100)
  expect_equal(nrow(result$test), 20)
  expect_equal(nrow(result$train), 80)

  # test for intersections
  expect_length(intersect(result$train$id, result$test$id), 0)
})

test_that("splitDataset_ throws error for invalid test_ratio", {
  df <- data.frame(id = 1:10)
  expect_error(splitDataset_(df, test_ratio = -0.1), "test_ratio")
  expect_error(splitDataset_(df, test_ratio = 1.5), "test_ratio")
  expect_error(splitDataset_(df, test_ratio = "1.5"), "test_ratio")
})

test_that("error when data is not a data.frame in splitDataset_", {
  expect_error(splitDataset_(42, test_ratio = 0.1), "'data' must be a data frame.")
})

test_that("splitDataset_ throws error for invalid seed", {
  df <- data.frame(id = 1:10)
  expect_error(splitDataset_(df, seed = "abc"), "seed")
  expect_error(splitDataset_(df, seed = c(1, 2)), "seed")
  expect_error(splitDataset_(df, seed = NA), "seed")
})

test_that("splitDataset_ throws error for empty data frame", {
  empty_df <- data.frame()
  expect_error(splitDataset_(empty_df), "must not be empty")
})

