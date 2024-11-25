set.seed(123)
data <- read.table("a24_clas_app.txt", header = TRUE, sep=" ")
data <- data[sample(nrow(data)),]
data$y <- as.factor(data$y)
# Create copies of original data
data_with_outliers <- data
data_cleaned <- data

# Define variable groups
continuous_vars <- paste0("X", 1:45)
ordinal_vars <- paste0("X", 46:50)

# Remove outliers for continuous variables only
for (var in continuous_vars) {
  Q1 <- quantile(data_cleaned[[var]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data_cleaned[[var]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  
  # Remove rows with outliers in any continuous variable
  data_cleaned <- data_cleaned[!(data_cleaned[[var]] < lower_bound | data_cleaned[[var]] > upper_bound), ]
}

# Calculate the ratio of rows deleted
cat("Rows deleted due to outliers:", (1 - nrow(data_cleaned) / nrow(data)) * 100, "%\n")

# Scale both datasets
# Dataset with outliers
preprocess_params_with_outliers <- preProcess(data_with_outliers[, continuous_vars], method = c("center", "scale"))
data_with_outliers_scaled <- data_with_outliers
data_with_outliers_scaled[, continuous_vars] <- predict(preprocess_params_with_outliers, data_with_outliers[, continuous_vars])

# Dataset without outliers
preprocess_params_cleaned <- preProcess(data_cleaned[, continuous_vars], method = c("center", "scale"))
data_cleaned_scaled <- data_cleaned
data_cleaned_scaled[, continuous_vars] <- predict(preprocess_params_cleaned, data_cleaned[, continuous_vars])

# Scale ordinal variables separately for both datasets
preprocess_params_ordinal_with_outliers <- preProcess(data_with_outliers[, ordinal_vars], method = c("center", "scale"))
data_with_outliers_scaled[, ordinal_vars] <- predict(preprocess_params_ordinal_with_outliers, data_with_outliers[, ordinal_vars])

preprocess_params_ordinal_cleaned <- preProcess(data_cleaned[, ordinal_vars], method = c("center", "scale"))
data_cleaned_scaled[, ordinal_vars] <- predict(preprocess_params_ordinal_cleaned, data_cleaned[, ordinal_vars])

# Store all versions in a list
datasets <- list(
  original_unscaled = data_with_outliers,
  original_scaled = data_with_outliers_scaled,
  clean_unscaled = data_cleaned,
  clean_scaled = data_cleaned_scaled
)
evaluate_base_model <- function(predictions_prob, true_labels) {
    predictions <- colnames(predictions_prob)[max.col(predictions_prob)]
    levels <- levels(true_labels)
    metrics <- multiClassSummary(data.frame(pred = predictions, obs = true_labels), lev = levels)
    return(list(
        Accuracy = metrics["Accuracy"],
        F1 = metrics["Mean_F1"]
    ))
}

evaluate_final_model <- function(predictions, true_labels) {
    levels <- levels(true_labels)
    metrics <- multiClassSummary(data.frame(pred = predictions, obs = true_labels), lev = levels)
    return(list(
        Accuracy = metrics["Accuracy"],
        F1 = metrics["Mean_F1"]
    ))
}

get_qda_predictions <- function(model, newdata) {
    pred_probs <- predict(model, newdata)$posterior
    return(pred_probs)
}

get_rf_predictions <- function(model, newdata) {
    pred_probs <- predict(model, newdata, type = "prob")
    return(pred_probs)
}

sample_data_subset <- function(data, target_col, method = "none", unscaled=FALSE) {
    if (method == "smote") {
        data_balanced <- SmoteClassif(as.formula(paste(target_col, "~ .")), dat = data, C.perc = "balance")
        if (unscaled){
            ordinal_cols <- grep("^X[4-5][0-9]$", names(data), value = TRUE)
            if(length(ordinal_cols) > 0) {
                data_balanced[,ordinal_cols] <- lapply(data_balanced[,ordinal_cols], round)
            }
        }
    } else if (method == "up") {
        data_balanced <- RandOverClassif(as.formula(paste(target_col, "~ .")), dat = data, C.perc = "balance")
        if (unscaled){
            ordinal_cols <- grep("^X[4-5][0-9]$", names(data), value = TRUE)
            if(length(ordinal_cols) > 0) {
                data_balanced[,ordinal_cols] <- lapply(data_balanced[,ordinal_cols], round)
            }
        }
    } else {
        data_balanced <- data
    }
    return(data_balanced)
}

get_kernel_features <- function(x, centers, sigma = 1) {
    kernels <- apply(centers, 1, function(c) {
        exp(-rowSums((sweep(x, 2, c))^2) / (2 * sigma^2))
    })
    return(kernels)
}

get_gmm_predictions <- function(model, newdata) {
    kernel_features <- get_kernel_features(newdata, model$centers, model$sigma)
    probs <- predict(model$gmm, kernel_features, what = "z")
    return(probs)
}

train_kernel_gmm <- function(data, G = 3, kernel_sigma = 1) {
    scaled_data <- scale(data)
    init_clusters <- kmeans(scaled_data, centers = G)
    centers <- init_clusters$centers
    kernel_features <- get_kernel_features(scaled_data, centers, kernel_sigma)
    gmm_model <- Mclust(kernel_features, G = G)
    return(list(
        gmm = gmm_model,
        centers = centers,
        sigma = kernel_sigma
    ))
}

combine_two_predictions <- function(pred1, pred2, prefix1, prefix2) {
    df1 <- as.data.frame(pred1)
    names(df1) <- paste0(prefix1, "_", 1:ncol(df1))
    
    df2 <- as.data.frame(pred2)
    names(df2) <- paste0(prefix2, "_", 1:ncol(df2))
    
    combined_df <- cbind(df1, df2)
    return(as.data.frame(combined_df))
}

# Initialize results dataframe
stacked_results <- data.frame(
    Stack_Pattern = character(),
    Dataset = character(),
    Sampling_Method = character(),
    Fold = integer(),
    Accuracy = double(),
    F1 = double()
)

# Define variable groups
gaussian_vars <- paste0("X", 21:45)
skewed_vars <- paste0("X", 1:20)
ordinal_vars <- paste0("X", 46:50)

# Define all patterns to test
patterns <- list(
    # Original patterns
    RF_QDA_RF = list(skewed = "RF", gaussian = "QDA", ordinal = "RF"),
    GMM_QDA_GMM = list(skewed = "GMM", gaussian = "QDA", ordinal = "GMM"),
    RF_QDA_GMM = list(skewed = "RF", gaussian = "QDA", ordinal = "GMM"),
    GMM_QDA_RF = list(skewed = "GMM", gaussian = "QDA", ordinal = "RF"),
    
    # New patterns without ordinal variables
    GMM_QDA = list(skewed = "GMM", gaussian = "QDA", ordinal = NULL),
    RF_QDA = list(skewed = "RF", gaussian = "QDA", ordinal = NULL),
    
    # New patterns without skewed variables
    QDA_GMM = list(skewed = NULL, gaussian = "QDA", ordinal = "GMM"),
    QDA_RF = list(skewed = NULL, gaussian = "QDA", ordinal = "RF")
)

# Parameters
sampling_methods <- c("none", "smote", "up")
set.seed(123)

# Main testing loop
for (pattern_name in names(patterns)) {
    pattern <- patterns[[pattern_name]]
    
    for (dataset_name in names(datasets)) {
        flag_unscaled <- dataset_name %in% c("original_unscaled", "clean_unscaled")
        dataset <- datasets[[dataset_name]]
        
        # Create 5 folds
        folds <- createFolds(dataset$y, k = 5, list = TRUE, returnTrain = FALSE)
        
        for (fold_idx in 1:5) {
            test_indices <- folds[[fold_idx]]
            data_test <- dataset[test_indices, ]
            data_train <- dataset[-test_indices, ]
            
            for (sampling_method in sampling_methods) {
                print(paste("Pattern:", pattern_name))
                print(paste("Dataset:", dataset_name))
                print(paste("  Fold:", fold_idx))
                print(paste("    Sampling method:", sampling_method))
                flush.console()
                
                tryCatch({
                    # Apply sampling
                    sampled_train <- sample_data_subset(data_train, "y", sampling_method, unscaled = flag_unscaled)
                    
                    # Initialize prediction variables
                    train_skewed_pred <- NULL
                    test_skewed_pred <- NULL
                    train_ord_pred <- NULL
                    test_ord_pred <- NULL
                    
                    # 1. Train and get predictions for skewed variables (if included in pattern)
                    if (!is.null(pattern$skewed)) {
                        if (pattern$skewed == "RF") {
                            skewed_model <- randomForest(y ~ ., data = sampled_train[, c("y", skewed_vars)],
                                                       ntree = 500, mtry = ceiling(sqrt(length(skewed_vars))),nodesize = 5)
                            train_skewed_pred <- get_rf_predictions(skewed_model, sampled_train[, skewed_vars])
                            test_skewed_pred <- get_rf_predictions(skewed_model, data_test[, skewed_vars])
                        } else {
                            skewed_matrix_train <- as.matrix(sampled_train[, skewed_vars])
                            skewed_matrix_test <- as.matrix(data_test[, skewed_vars])
                            skewed_model <- train_kernel_gmm(skewed_matrix_train, G = 3)
                            train_skewed_pred <- get_gmm_predictions(skewed_model, skewed_matrix_train)
                            test_skewed_pred <- get_gmm_predictions(skewed_model, skewed_matrix_test)
                        }
                    }
                    
                    # 2. QDA for Gaussian variables (always included)
                    qda_model <- qda(y ~ ., data = sampled_train[, c("y", gaussian_vars)])
                    train_qda_pred <- get_qda_predictions(qda_model, sampled_train[, gaussian_vars])
                    test_qda_pred <- get_qda_predictions(qda_model, data_test[, gaussian_vars])
                    
                    # 3. Train and get predictions for ordinal variables (if included in pattern)
                    if (!is.null(pattern$ordinal)) {
                        if (pattern$ordinal == "RF") {
                            ord_model <- randomForest(y ~ ., data = sampled_train[, c("y", ordinal_vars)],
                                                    ntree = 500, mtry = ceiling(sqrt(length(ordinal_vars))),nodesize = 5)
                            train_ord_pred <- get_rf_predictions(ord_model, sampled_train[, ordinal_vars])
                            test_ord_pred <- get_rf_predictions(ord_model, data_test[, ordinal_vars])
                        } else {
                            ord_matrix_train <- as.matrix(sampled_train[, ordinal_vars])
                            ord_matrix_test <- as.matrix(data_test[, ordinal_vars])
                            ord_model <- train_kernel_gmm(ord_matrix_train, G = 3)
                            train_ord_pred <- get_gmm_predictions(ord_model, ord_matrix_train)
                            test_ord_pred <- get_gmm_predictions(ord_model, ord_matrix_test)
                        }
                    }
                    
                    # Combine predictions based on pattern
                    if (!is.null(pattern$skewed) && !is.null(pattern$ordinal)) {
                        # Full three-model stack
                        meta_features_train <- combine_predictions(
                            train_qda_pred,
                            train_skewed_pred,
                            train_ord_pred
                        )
                        meta_features_test <- combine_predictions(
                            test_qda_pred,
                            test_skewed_pred,
                            test_ord_pred
                        )
                    } else if (!is.null(pattern$skewed)) {
                        # Two-model stack with skewed
                        meta_features_train <- combine_two_predictions(
                            train_qda_pred,
                            train_skewed_pred,
                            "qda",
                            "skewed"
                        )
                        meta_features_test <- combine_two_predictions(
                            test_qda_pred,
                            test_skewed_pred,
                            "qda",
                            "skewed"
                        )
                    } else {
                        # Two-model stack with ordinal
                        meta_features_train <- combine_two_predictions(
                            train_qda_pred,
                            train_ord_pred,
                            "qda",
                            "ord"
                        )
                        meta_features_test <- combine_two_predictions(
                            test_qda_pred,
                            test_ord_pred,
                            "qda",
                            "ord"
                        )
                    }
                    
                    # Train meta-learner
                    meta_learner <- randomForest(
                        x = meta_features_train,
                        y = sampled_train$y,
                        ntree = 500,
                        mtry = ceiling(sqrt(ncol(meta_features_train))),
                        nodesize = 5
                    )
                    
                    # Final predictions
                    final_predictions <- predict(meta_learner, newdata = meta_features_test)
                    
                    # Evaluate
                    test_metrics <- evaluate_final_model(final_predictions, data_test$y)
                    
                    # Store results
                    stacked_results <- rbind(stacked_results, data.frame(
                        Stack_Pattern = pattern_name,
                        Dataset = dataset_name,
                        Sampling_Method = sampling_method,
                        Fold = fold_idx,
                        Accuracy = test_metrics$Accuracy,
                        F1 = test_metrics$F1
                    ))
                    
                }, error = function(e) {
                    print(paste("Error in pattern", pattern_name, ":", e$message))
                })
            }
        }
    }
}

# Calculate mean metrics across folds
stacked_summary <- aggregate(
    cbind(Accuracy, F1) ~ Stack_Pattern + Dataset + Sampling_Method,
    data = stacked_results,
    FUN = mean
)

# Sort by accuracy
stacked_summary <- stacked_summary[order(-stacked_summary$Accuracy), ]

# Print results
print("Complete summary of all combinations (sorted by accuracy):")
print(stacked_summary)