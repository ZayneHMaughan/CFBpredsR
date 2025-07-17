#####
library(caret)
library(themis)
library(xgboost)
library(Metrics)
library(recipes)


train_data <- modeling_data |>
  dplyr::filter(!season.x %in% c(2023, 2024)) |>
  select(-c(conference.x, conference.y))

test_data <- modeling_data |>
  dplyr::filter(season.x %in% c(2023, 2024)) |>
  dplyr::select(-c(conference.x, conference.y))


######################## SMOTE FOR THE DATA AND THE MODEL ######################
smote_recipe <- recipe(playoff ~ ., data = train_data) |>
  update_role(team, new_role = "id") |>
  # step_rm(off_ppa, off_total_ppa, off_field_pos_avg_predicted_points,
  #         off_havoc_total,off_success_rate, off_standard_downs_rate,
  #         off_rushing_plays_ppa, off_rushing_plays_rate,off_drives,
  #         def_drives, def_ppa, def_field_pos_avg_predicted_points,
  #         def_havoc_total, def_success_rate,def_standard_downs_rate,
  #         def_rushing_plays_ppa, def_rushing_plays_rate,
  #         def_passing_downs_total_ppa, def_passing_plays_ppa, season.x,season.y,
  #         pass_atts, rush_yds, first_downs, penalty_yds, kick_returns,
  #         def_standard_downs_ppa, def_open_field_yds, off_line_yds_total,
  #         net_pass_yds, off_open_field_yds, off_passing_plays_ppa,
  #         def_standard_downs_success_rate,off_passing_downs_explosiveness,
  #         fumbles_lost,
  #         off_explosiveness, def_passing_downs_explosiveness,
  #         def_explosiveness, def_passing_downs_explosiveness, fourth_downs,
  #         def_havoc_front_seven, fourth_down_convs, off_passing_plays_explosiveness,
  #         off_rushing_plays_explosiveness, off_standard_downs_explosiveness,
  #         def_rushing_plays_explosiveness,kick_return_yds,
  #         def_standard_downs_explosiveness, pass_comps, third_downs, penalties,
  #         punt_return_TDs, def_havoc_db, off_havoc_front_seven ) |>
  step_smote(playoff)

# Prep only once on training data
smote_prep <- prep(smote_recipe, retain = TRUE)

# Apply to training data
smote_train <- juice(smote_prep)

# Apply the same transformations to test data
smote_test <- bake(smote_prep, new_data = test_data)


###################### Model ##########################

library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

xgb_grid <- expand.grid(
  nrounds = c(100,200),
  max_depth = c(4,6),
  eta = c(0.1, 1),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 1),
  min_child_weight = c(1,2),
  subsample = c(0.7, 0.9)
)

set.seed(24)
repeated_oob <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


model <- train(playoff ~ .,
               data = smote_train |> select(-team),
               trControl = repeated_oob,
               method = "xgbTree",
               metric = "ROC",
               tuneGrid = xgb_grid
)

model$bestTune

stopCluster(cl)
registerDoSEQ()



###################### SVMs ###################################################


library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

svm_grid <- expand.grid(
  C = c(0.25, 0.5, 1),
  sigma = c(0.01, 0.015, 0.02)
)

set.seed(24)

repeated_oob <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

svm_model <- train(playoff ~ .,
                   data = smote_train |> select(-team),
                   trControl = repeated_oob,
                   method = "svmRadial",
                   metric = "ROC",
                   tuneGrid = svm_grid
)
svm_model$bestTune

stopCluster(cl)
registerDoSEQ()

##################### Random Forest ############################################
library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)



repeated_oob <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8),
  splitrule = "gini",
  min.node.size = 1
)

set.seed(24)
rf_model <- train(playoff ~ .,
                  data = smote_train |> select(-team),
                  trControl = repeated_oob,
                  method = "ranger",  # <-- important change here
                  metric = "ROC",
                  tuneGrid = rf_grid,
                  importance = 'impurity'
)
rf_model$bestTune

stopCluster(cl)
registerDoSEQ()

################################ KNN
library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

repeated_oob <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

knn_grid <- expand.grid(
  k = c(3, 5, 7, 9, 11)
)

set.seed(24)
knn_model <- train(playoff ~ .,
                   data = smote_train |> select(-team),
                   trControl = repeated_oob,
                   method = "knn",
                   metric = "ROC",
                   tuneGrid = knn_grid
)


stopCluster(cl)
registerDoSEQ()


########################### Logistic Regression ##############################

# library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

repeated_oob <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

lasso_grid <- expand.grid(
  alpha = 1,                     # 1 = LASSO penalty
  lambda = c(0.0001, 0.001, 0.01, 0.1, 1)  # different levels of shrinkage
)

set.seed(24)
lasso_model <- train(playoff ~ .,
                     data = smote_train |> select(-team),
                     trControl = repeated_oob,
                     method = "glmnet",
                     metric = "ROC",
                     tuneGrid = lasso_grid
)

stopCluster(cl)
registerDoSEQ()


############################ Decision Tree ####################################
# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

cart_grid <- expand.grid(
  cp = c(0.001, 0.01, 0.05, 0.1)
)

set.seed(24)
cart_model <- train(playoff ~ .,
                    data = smote_train |> select(-team),
                    trControl = repeated_oob,
                    method = "rpart",
                    metric = "ROC",
                    tuneGrid = cart_grid
)


stopCluster(cl)
registerDoSEQ()


# #################### ADA Boost #########################
# cl <- makeCluster(parallel::detectCores() - 1)
# registerDoParallel(cl)
#
# # AdaBoost tuning grid (for adaboost_m1)
# adaboost_grid <- expand.grid(
#   nIter = c(50, 100, 150),  # number of boosting rounds
#   maxdepth = c(1, 2, 3)     # max tree depth
# )
#
# # Train AdaBoost model using adaboost_m1
# set.seed(24)
# adaboost_model <- train(playoff ~ .,
#                         data = smote_train |> select(-team),
#                         trControl = repeated_oob,
#                         method = 'AdaBoost.M1',   # <-- NOT "adaboost" anymore
#                         metric = "ROC",
#                         tuneGrid = adaboost_grid)
# stopCluster(cl)
# registerDoSEQ()
#

##################
model_list <- c(model, svm_model, rf_model, knn_model, lasso_model, cart_model)

bind_rows(
  get_metrics(model, smote_test, playoff, "XGboost"),
  get_metrics(svm_model, smote_test,playoff, "SVM" ),
  get_metrics(rf_model, smote_test,playoff, "Random Forest" ),
  get_metrics(knn_model, smote_test,playoff, "KNN" ),
  get_metrics(lasso_model, smote_test,playoff, "LASSO" ),
  get_metrics(cart_model, smote_test,playoff, "Decision Tree" )

)

varImp(model)
varImp(rf_model)


get_metrics(model, smote_test, playoff)
