#'
#'
#'
#' @export
#'
library(Metrics)
get_metrics <- function(model, newdata, response, model_name) {

  preds_class <- predict(model, newdata)
  preds_prob  <- predict(model, newdata, type = "prob")[, "True"]

  truth <- newdata[[deparse(substitute(response))]]

  tibble(
    Model = model_name,
    Accuracy  = accuracy_vec(truth = truth, estimate = preds_class),
    Precision = precision_vec(truth = truth, estimate = preds_class),
    Recall    = recall_vec(truth = truth, estimate = preds_class),
    AUC       = roc_auc_vec(truth = truth, estimate = preds_prob, event_level = "second")
  )
}
