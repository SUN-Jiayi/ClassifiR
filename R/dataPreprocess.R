#' Preprocess a Data Frame for Machine Learning
#'
#' This function cleans and preprocesses a dataset by removing missing values,
#' optionally splitting a `Date` column into `Year`, `Month`, and `Day`,
#' and performing one-hot encoding for categorical variables. It also allows
#' specifying a label column which will be moved to the last column of the returned data.
#'
#' @param data A `data.frame`. The input dataset to be preprocessed.
#' @param one_hot Logical, default is `TRUE`. Whether to apply one-hot encoding to character and factor columns.
#' @param split_date Logical, default is `TRUE`. Whether to split a `Date` column into `Year`, `Month`, and `Day`.
#' @param label_col Character, default is `NULL`. The name of the label (target) column. This column will be moved to the last position.
#'
#' @return A preprocessed `data.frame` with missing rows removed, optional one-hot encoding applied,
#'         and the label column moved to the end. If `split_date = TRUE`, the `Date` column is removed
#'         and replaced with three columns: `Year`, `Month`, and `Day`.
#'
#' @details
#' The function performs several preprocessing steps in the following order:
#' \enumerate{
#'   \item Asserts that `data` is a data frame.
#'   \item Removes all rows containing any `NA` values.
#'   \item If `split_date` is `TRUE`, attempts to parse the `Date` column and split it into `Year`, `Month`, and `Day`.
#'   \item If `one_hot` is `TRUE`, one-hot encodes all categorical (character or factor) columns, except for the label column.
#'   \item The label column, if specified, is moved to the end of the dataset.
#' }
#'
#' If `label_col` is a categorical variable and `one_hot = TRUE`, then it must produce exactly one column during one-hot encoding.
#'
#' @import assertthat
#' @importFrom lubridate ymd year month day
#'
#' @examples
#' # Example with categorical variable and label column
#' df <- data.frame(A = c("Yes", "No", "Yes"), B = 1:3)
#' dataPreprocessing(df, one_hot = TRUE, split_date = FALSE, label_col = "A")
#'
#' # Example with date splitting
#' df2 <- data.frame(Date = c("2020-01-01", "2021-03-15"), X = 1:2)
#' dataPreprocessing(df2, one_hot = FALSE, split_date = TRUE)
#'
#' # Example with NA removal
#' df3 <- data.frame(A = c(1, NA), B = c("x", "y"))
#' \dontrun{
#' dataPreprocessing(df3)  # Throws an error due to all rows removed
#' }
#'
#' @seealso \code{\link[assertthat]{assert_that}}, \code{\link[lubridate]{ymd}}
#'
#' @export
dataPreprocessing <- function(data, one_hot = TRUE, split_date = TRUE, label_col = NULL){
  assertthat::assert_that(is.data.frame(data), msg = "'data' must be a data frame.")
  if (!is.null(label_col)) {
    assertthat::assert_that(label_col %in% colnames(data),
                            msg = paste("label_col does not exist in the dataset."))
  }
  data <- na.omit(data)
  assertthat::assert_that(nrow(data) > 0,
                          msg = "All rows were removed after omitting NA values. Please check your data.")
  if (split_date && "Date" %in% colnames(data)) {
    # convert into Date
    date_vec <- suppressWarnings(lubridate::ymd(data$Date))
    assertthat::assert_that(all(!is.na(date_vec)),
                            msg = "Some entries in the 'Date' column could not be parsed. Please check the format.")
    data$Year <- lubridate::year(date_vec)
    data$Month <- lubridate::month(date_vec)
    data$Day <- lubridate::day(date_vec)

    # Delete the original Date column
    data$Date <- NULL
  } else if (split_date){
    assertthat::assert_that(FALSE, msg = "No 'Date' column found in the dataset when 'split_date' is TRUE.")
  }
  label_vector <- NULL
  label_found <- FALSE
  if (one_hot) {
    categorical_cols <- names(Filter(function(col) is.character(col) || is.factor(col), data))

    if (length(categorical_cols) > 0) {
      # save the non categorical data
      non_cat_data <- data[ , !(names(data) %in% categorical_cols), drop = FALSE]

      # construct the formula
      formula <- as.formula(paste("~", paste(categorical_cols, collapse = " + ")))
      dummies <- as.data.frame(model.matrix(formula, data = data)[, -1, drop = FALSE])

      # If the label column is categorical, extract the dummy as the final label
      if (!is.null(label_col) && label_col %in% categorical_cols) {
        label_dummy_col <- grep(paste0("^", label_col), names(dummies), value = TRUE)
        assertthat::assert_that(length(label_dummy_col) == 1,
                                msg = "Label column after one-hot encoding must result in exactly one column.")
        label_vector <- dummies[[label_dummy_col]]
        names(label_vector) <- NULL
        dummies <- dummies[, !names(dummies) %in% label_dummy_col, drop = FALSE]
        label_found <- TRUE
      }

      # combine the non categotical data and the one hot encoded data
      data <- cbind(non_cat_data, dummies)
    }
  }

  # if the label column is numeric or one-hot is FALSE
  if (!label_found && !is.null(label_col)) {
    label_vector <- data[[label_col]]
    data[[label_col]] <- NULL
  }

  # add the label to the last column
  if (!is.null(label_col)) {
    data[[label_col]] <- label_vector
  }
  return(data)
}
#' Split a Data Frame into Training and Testing Sets
#'
#' This function splits a data frame into a training set and a testing set
#' based on a user-defined ratio. The split is randomized but reproducible using a seed.
#'
#' @param data A non-empty data frame.
#' @param test_ratio A numeric value between 0 and 1. Represents the proportion of test data. Default is 0.2.
#' @param seed An integer random seed to ensure reproducibility. Default is 42.
#'
#' @return A list with two data frames:
#' \describe{
#'   \item{train}{Training data frame.}
#'   \item{test}{Testing data frame.}
#' }
#'
#' @examples
#' df <- data.frame(x = 1:10, y = 11:20)
#' result <- splitDataset_(df, test_ratio = 0.3, seed = 123)
#' nrow(result$train) + nrow(result$test) == nrow(df)
#'
#' @export
splitDataset_ <- function(data, test_ratio = 0.2, seed = 42) {
  assertthat::assert_that(is.data.frame(data), msg = "'data' must be a data frame.")
  assertthat::assert_that(nrow(data) > 0, msg = "'data' must not be empty.")
  assertthat::assert_that(test_ratio > 0 && test_ratio < 1, msg = "'test_ratio' must be between 0 and 1.")
  assertthat::assert_that(
    is.numeric(seed) && length(seed) == 1 && seed == as.integer(seed),
    msg = "'seed' must be a single integer value."
  )
  set.seed(seed)
  n <- nrow(data)
  test_indices <- sample(seq_len(n), size = floor(n * test_ratio))
  train_indices <- setdiff(seq_len(n), test_indices)

  train_data <- data[train_indices, , drop = FALSE]
  test_data <- data[test_indices, , drop = FALSE]

  return(list(train = train_data, test = test_data))
}
# data("weatherAUS")
# clean <- dataPreprocessing(weatherAUS, label_col = "RainTomorrow")
# # head(w)
# # dim(w)

