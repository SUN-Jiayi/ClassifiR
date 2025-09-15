#' Evaluate Binary Classification Model
#'
#' @description
#' Computes key performance metrics for binary classification, including accuracy, precision, recall, F1 score, specificity, balanced accuracy,
#' Matthews correlation coefficient (MCC), log loss, and the confusion matrix. This function is useful for evaluating classifiers with predicted
#' labels and optional predicted probabilities.
#'
#' @param y_true True labels (vector of 0/1).
#' @param y_pred Predicted labels (vector of 0/1).1
#' @param y_prob Optional predicted probabilities (vector with values in [0, 1])
#' @return A named list containing the following components:
#' \describe{
#'   \item{accuracy}{Proportion of correct predictions}
#'   \item{precision}{Proportion of predicted positives that are actually positive}
#'   \item{recall}{Proportion of actual positives that are correctly predicted}
#'   \item{specificity}{Proportion of actual negatives that are correctly predicted}
#'   \item{balanced_accuracy}{Average of recall and specificity}
#'   \item{f1}{Harmonic mean of precision and recall}
#'   \item{mcc}{Matthews Correlation Coefficient, a balanced metric even for imbalanced classes}
#'   \item{log_loss}{Logarithmic loss based on predicted probabilities (only if \code{y_prob} is provided)}
#'   \item{confusion_matrix}{A 2x2 matrix showing predicted vs actual values}
#' }
#'
#' @examples
#' y_true = c(0, 1, 1, 0, 1, 0, 1, 0)
#' y_pred = c(0, 1, 1, 0, 0, 0, 1, 1)
#' y_prob = c(0.1, 0.9, 0.85, 0.2, 0.4, 0.3, 0.8, 0.7)
#' evaluate_binary_classification(y_true, y_pred, y_prob)
#'
#' @export
evaluate_binary_classification = function(y_true, y_pred, y_prob = NULL) {
  assertthat::assert_that(
    is.numeric(y_true),
    all(y_true %in% c(0, 1)),
    msg = "y_true must be a numeric vector containing only 0 and 1."
  )

  assertthat::assert_that(
    is.numeric(y_pred),
    all(y_pred %in% c(0, 1)),
    msg = "y_pred must be a numeric vector containing only 0 and 1."
  )

  assertthat::assert_that(
    length(y_true) == length(y_pred),
    msg = "y_true and y_pred must have the same length."
  )

  if (!is.null(y_prob)) {
    assertthat::assert_that(
      is.numeric(y_prob),
      all(!is.na(y_prob)),
      all(y_prob >= 0 & y_prob <= 1),
      length(y_prob) == length(y_true),
      msg = "y_prob must be a numeric vector in [0, 1] with same length as y_true."
    )
  }

  TP = as.numeric(sum(y_true == 1 & y_pred == 1))
  TN = as.numeric(sum(y_true == 0 & y_pred == 0))
  FP = as.numeric(sum(y_true == 0 & y_pred == 1))
  FN = as.numeric(sum(y_true == 1 & y_pred == 0))

  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall = ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1 = ifelse(is.na(precision) || is.na(recall) || (precision + recall) == 0, NA,
              2 * precision * recall / (precision + recall))
  specificity = ifelse((TN + FP) == 0, NA, TN / (TN + FP))
  balanced_accuracy = mean(c(recall, specificity), na.rm = TRUE)

  denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc = ifelse(denominator == 0, NA, (TP * TN - FP * FN) / denominator)

  if (!is.null(y_prob)) {
    assertthat::assert_that(length(y_prob) == length(y_true))
    epsilon = 1e-15
    y_prob = pmin(pmax(y_prob, epsilon), 1 - epsilon)
    log_loss = -mean(y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob))
  } else {
    log_loss = NA
  }

  confusion_matrix = matrix(c(TP, FP, FN, TN), nrow = 2, byrow = TRUE,
                            dimnames = list(Predicted = c("1", "0"), Actual = c("1", "0")))

  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    specificity = specificity,
    balanced_accuracy = balanced_accuracy,
    f1 = f1,
    mcc = mcc,
    log_loss = log_loss,
    confusion_matrix = confusion_matrix
  ))
}


#' Evaluate ROC and AUC
#'
#' @description
#' Computes the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) for a binary classification model.
#' This is useful for evaluating the ranking performance of a model based on predicted probabilities.
#'
#' Internally uses the \code{pROC} package to calculate and return both the ROC curve object and its corresponding AUC.
#'
#' @param y_true True labels (0/1).
#' @param y_prob Predicted probabilities (0~1).
#' @return A list containing:
#' \describe{
#'   \item{roc}{An object of class `roc` from the \code{pROC} package, containing coordinates and thresholds}
#'   \item{auc}{The AUC value (numeric)}
#' }
#'
#' @examples
#' y_true = c(0, 1, 1, 0, 1, 0, 0, 1)
#' y_prob = c(0.2, 0.9, 0.8, 0.3, 0.7, 0.1, 0.4, 0.95)
#' result = evaluate_roc_auc(y_true, y_prob)
#' print(result$auc)
#' plot(result$roc)
#'
#' @export
evaluate_roc_auc = function(y_true, y_prob) {
  assertthat::assert_that(
    is.numeric(y_true),
    all(y_true %in% c(0, 1)),
    msg = "y_true must be a numeric vector containing only 0 and 1."
  )

  assertthat::assert_that(
    is.numeric(y_prob),
    all(!is.na(y_prob)),
    all(y_prob >= 0 & y_prob <= 1),
    msg = "y_prob must be a numeric vector with values in [0, 1]."
  )

  assertthat::assert_that(
    length(y_true) == length(y_prob),
    msg = "y_true and y_prob must have the same length."
  )

  roc_obj = pROC::roc(y_true, y_prob)
  auc_value = pROC::auc(roc_obj)

  return(list(
    roc = roc_obj,
    auc = auc_value
  ))
}


#' Plot ROC Curves for Multiple Models
#'
#' @description
#' Plots ROC curves for multiple classification models on the same ggplot.
#' AUC values are displayed in the legend, and true labels are compared against predicted probabilities.
#'
#' @param y_true A binary vector of true labels (0/1).
#' @param prob_list A named list of numeric vectors.
#'   Each element corresponds to a model and must contain predicted probabilities for class 1.
#'   The name of each element will be used as the model name in the legend.
#' @param title Optional title for the plot.
#'
#' @return A ggplot object showing all ROC curves and AUCs.
#'
#' @examples
#' y = c(0, 1, 1, 0, 1, 0, 1, 0)
#' pred_list = list(
#'   DecisionTree = c(0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4),
#'   RandomForest = c(0.1, 0.9, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3)
#' )
#' plot_model_roc_comparison(y, pred_list)
#'
#' @export
plot_model_roc_comparison = function(y_true, prob_list, title = "ROC Curve Comparison") {
  assertthat::assert_that(
    is.numeric(y_true),
    all(y_true %in% c(0, 1)),
    length(y_true) >= 2,
    msg = "y_true must be a numeric vector of 0s and 1s, and contain at least 2 elements."
  )

  assertthat::assert_that(
    is.list(prob_list),
    !is.null(names(prob_list)),
    all(sapply(prob_list, is.numeric)),
    all(sapply(prob_list, function(x) all(!is.na(x)))),
    all(sapply(prob_list, function(x) all(x >= 0 & x <= 1))),
    all(sapply(prob_list, function(x) length(x) == length(y_true))),
    msg = "prob_list must be a named list of numeric vectors with values in [0,1] and same length as y_true."
  )

  assertthat::assert_that(
    is.character(title),
    length(title) == 1,
    msg = "title must be a single character string."
  )

  roc_data = do.call(rbind, lapply(names(prob_list), function(model_name) {
    result = evaluate_roc_auc(y_true, prob_list[[model_name]])
    roc_obj = result$roc
    auc_val = as.numeric(result$auc)

    coords = pROC::coords(roc_obj, "all", ret = c("specificity", "sensitivity"))
    data.frame(
      FPR = 1 - coords$specificity,
      TPR = coords$sensitivity,
      Model = sprintf("%s (AUC = %.3f)", model_name, auc_val)
    )
  }))


  ggplot2::ggplot(roc_data, ggplot2::aes(x = FPR, y = TPR, color = Model)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_abline(linetype = "dashed", color = "grey") +
    ggplot2::labs(
      title = title,
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Model"
    ) +
    ggplot2::theme_minimal()
}


#' Compare Multiple Binary Classification Models and Identify Best by Each Criterion
#'
#' @description
#' Evaluates multiple binary classifiers using standard performance metrics and identifies the best model for each metric.
#' Optionally supports both hard predictions (y_pred) and predicted probabilities (y_prob) for computing log loss.
#'
#' @param y_true A numeric vector of true binary labels (0/1).
#' @param pred_list A named list of numeric vectors.
#'   Each element contains predicted labels (0/1) for a model.
#'   The name of each element will be used to identify the model.
#' @param prob_list Optional named list of numeric vectors with predicted probabilities (values in [0, 1]) for each model.
#'   Required if you want to compute log loss for model comparison. The names must match pred_list.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{summary_table}{A data.frame summarizing evaluation metrics (Accuracy, Precision, Recall, Specificity, Balanced Accuracy, F1, MCC, Log Loss) for each model.}
#'   \item{best_models}{A named list indicating the best model for each individual metric.}
#' }
#'
#' @examples
#' y = c(0, 1, 1, 0, 1, 0, 1, 0)
#' preds = list(
#'   DecisionTree = c(0, 1, 1, 0, 1, 0, 0, 0),
#'   RandomForest = c(0, 1, 1, 0, 1, 0, 1, 0)
#' )
#' probs = list(
#'   DecisionTree = c(0.2, 0.9, 0.85, 0.1, 0.95, 0.3, 0.4, 0.5),
#'   RandomForest = c(0.1, 0.95, 0.87, 0.2, 0.97, 0.1, 0.88, 0.3)
#' )
#' result = compare_models(y, preds, prob_list = probs)
#' print(result$summary_table)
#' print(result$best_models)
#'
#' @export

compare_models = function(y_true, pred_list, prob_list = NULL) {
  assertthat::assert_that(
    is.numeric(y_true),
    all(y_true %in% c(0, 1)),
    length(y_true) >= 2,
    msg = "y_true must be a numeric vector of 0s and 1s with at least 2 elements."
  )

  assertthat::assert_that(
    is.list(pred_list),
    !is.null(names(pred_list)),
    all(sapply(pred_list, is.numeric)),
    all(sapply(pred_list, function(p) length(p) == length(y_true))),
    msg = "pred_list must be a named list of numeric vectors, each with the same length as y_true."
  )

  if (!is.null(prob_list)) {
    assertthat::assert_that(
      is.list(prob_list),
      !is.null(names(prob_list)),
      all(sapply(prob_list, is.numeric)),
      all(sapply(prob_list, function(p) length(p) == length(y_true))),
      all(sapply(prob_list, function(p) all(p >= 0 & p <= 1))),
      identical(sort(names(pred_list)), sort(names(prob_list))),
      msg = "prob_list must be a named list of numeric vectors in [0,1], same length as y_true, and with names matching pred_list."
    )
  }

  results = lapply(names(pred_list), function(name) {
    y_pred = pred_list[[name]]
    y_prob = if (!is.null(prob_list)) prob_list[[name]] else NULL

    metrics = evaluate_binary_classification(y_true, y_pred, y_prob)

    data.frame(
      Model = name,
      Accuracy = round(metrics$accuracy, 4),
      Precision = round(metrics$precision, 4),
      Recall = round(metrics$recall, 4),
      Specificity = round(metrics$specificity, 4),
      Balanced_Accuracy = round(metrics$balanced_accuracy, 4),
      F1 = round(metrics$f1, 4),
      MCC = round(metrics$mcc, 4),
      LogLoss = if (!is.null(metrics$log_loss)) round(metrics$log_loss, 4) else NA,
      stringsAsFactors = FALSE
    )
  })

  summary_table = do.call(rbind, results)

  best_models = list(
    Accuracy  = summary_table$Model[which.max(summary_table$Accuracy)],
    Precision = summary_table$Model[which.max(summary_table$Precision)],
    Recall    = summary_table$Model[which.max(summary_table$Recall)],
    Specificity = summary_table$Model[which.max(summary_table$Specificity)],
    Balanced_Accuracy = summary_table$Model[which.max(summary_table$Balanced_Accuracy)],
    F1 = summary_table$Model[which.max(summary_table$F1)],
    MCC = summary_table$Model[which.max(summary_table$MCC)],
    LogLoss = if (all(is.na(summary_table$LogLoss))) NA else summary_table$Model[which.min(summary_table$LogLoss)]
  )

  return(list(
    summary_table = summary_table,
    best_models = best_models
  ))
}

