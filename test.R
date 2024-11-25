# Required packages for regression
required_packages <- c(
  "caret",
  "MASS",
  "randomForest",
  "e1071",
  "ggplot2",
  "dplyr",
  "corrplot",
  "nnet",
  "mgcv",
  "devtools",
  "moments",
  "xgboost",
  "glmnet",
  "pls",
  "tidyr"
)

# Function to install and load packages
install_and_load_packages <- function(packages) {
  cat("Checking and installing required packages...\n")
  
  for (package in packages) {
    if (!require(package, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("Installing package: %s\n", package))
      install.packages(package, dependencies = TRUE)
      if (!require(package, character.only = TRUE, quietly = TRUE)) {
        stop(sprintf("Package '%s' installation failed", package))
      }
    } else {
      cat(sprintf("Package '%s' is already installed and loaded\n", package))
    }
  }
  cat("\nAll required packages are installed and loaded!\n\n")
}

# Install and load all required packages
install_and_load_packages(required_packages)

# Load data
set.seed(123)
data <- read.table("a24_reg_app.txt", header = TRUE, sep=" ")
data <- data[sample(nrow(data)),]

# Check for missing values
missing_values <- colSums(is.na(data))
missing_values <- missing_values[missing_values > 0]
if (length(missing_values) > 0) {
  cat("Missing values in the dataset:\n")
  print(missing_values)
} else {
  cat("No missing values in the dataset\n")
}

# Correlation analysis
cor_matrix <- cor(data[,1:100])  # X1-X100 are features

# Function to find high correlations
find_high_cors <- function(cor_matrix, threshold) {
  upper_tri <- upper.tri(cor_matrix)
  high_cors <- which(abs(cor_matrix) > threshold & upper_tri, arr.ind = TRUE)
  
  result <- data.frame(
    Var1 = rownames(cor_matrix)[high_cors[,1]],
    Var2 = colnames(cor_matrix)[high_cors[,2]],
    Correlation = cor_matrix[high_cors]
  )
  
  return(result[order(abs(result$Correlation), decreasing = TRUE),])
}

# Print correlations for different thresholds
thresholds <- c(0.9, 0.8, 0.7, 0.6)
for(threshold in thresholds) {
  high_cors <- find_high_cors(cor_matrix, threshold)
  if (nrow(high_cors) > 0) {
    cat("\nCorrelations above", threshold, ":\n")
    print(high_cors)
    cat("\nNumber of pairs:", nrow(high_cors), "\n")
  }
  else {
    cat("\nNo correlations above", threshold, "\n")
  }
}

# Distribution visualizations against target y
cat("\nCreating distribution plots against target variable...\n")

# Create data frame for plotting X variables against y
plot_data <- data[, 1:100]  # Get X variables
plot_data$y <- data$y       # Add y variable

# Convert to long format only for X variables
data_long <- tidyr::pivot_longer(
  plot_data,
  cols = starts_with("X"),
  names_to = "variable",
  values_to = "value"
)

# Create scatter plots with trend lines
p1 <- ggplot(data_long, aes(x = value, y = y)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  facet_wrap(~variable, 
             scales = "free_x", 
             ncol = 5) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 8),
    axis.text = element_text(size = 6),
    axis.title = element_text(size = 8)
  ) +
  labs(title = "Relationship between Features and Target",
       x = "Feature Value",
       y = "Target (y)")

# Display the scatter plots
print(p1)

# Calculate correlation with target and basic statistics
stats <- data_long %>%
  group_by(variable) %>%
  summarise(
    correlation_with_y = cor(value, y),
    mean = mean(value),
    sd = sd(value),
    skewness = moments::skewness(value),
    kurtosis = moments::kurtosis(value)
  ) %>%
  arrange(desc(abs(correlation_with_y)))

cat("\nFeature statistics and correlation with target:\n")
print(stats)

# Remove outliers using IQR method
cat("\nRemoving outliers using IQR method...\n")
data_cleaned <- data

for (var in paste0("X", 1:100)) {
  Q1 <- quantile(data_cleaned[[var]], 0.25)
  Q3 <- quantile(data_cleaned[[var]], 0.75)
  IQR_val <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  
  data_cleaned <- data_cleaned[data_cleaned[[var]] >= lower_bound & 
                              data_cleaned[[var]] <= upper_bound, ]
}

cat("Rows remaining after outlier removal:", nrow(data_cleaned), 
    "\nPercentage of data retained:", round(nrow(data_cleaned)/nrow(data)*100, 2), "%\n")

# Preprocessing function with PCA option
preprocess_data <- function(data, use_pca = FALSE, pca_threshold = 0.95) {
  # Split features and target
  X <- data[, 1:100]  
  y <- data[, 101]    
  
  # Scale features
  X_scaled <- scale(X)
  y_scaled <- scale(y)
  
  if (use_pca) {
    # Perform PCA
    pca_result <- prcomp(X_scaled)
    var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
    n_components <- which(var_explained >= pca_threshold)[1]
    
    X_processed <- pca_result$x[, 1:n_components]
    
    attr(X_processed, "pca_params") <- list(
      rotation = pca_result$rotation[, 1:n_components],
      center = attr(X_scaled, "scaled:center"),
      scale = attr(X_scaled, "scaled:scale")
    )
    
    cat("Number of PCA components used:", n_components, "\n")
    cat("Variance explained:", round(var_explained[n_components] * 100, 2), "%\n")
  } else {
    X_processed <- X_scaled
    attr(X_processed, "scaling") <- list(
      center = attr(X_scaled, "scaled:center"),
      scale = attr(X_scaled, "scaled:scale")
    )
  }
  
  attr(y_scaled, "scaling") <- list(
    center = attr(y_scaled, "scaled:center"),
    scale = attr(y_scaled, "scaled:scale")
  )
  
  return(list(X = X_processed, y = y_scaled))
}

# Function to descale predictions
descale_predictions <- function(scaled_predictions, scaling_params) {
  return(scaled_predictions * scaling_params$scale + scaling_params$center)
}

# Model evaluation function
evaluate_model <- function(predictions, true_values) {
  mse <- mean((predictions - true_values)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions - true_values))
  r2 <- 1 - sum((true_values - predictions)^2) / sum((true_values - mean(true_values))^2)
  
  return(list(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R2 = r2
  ))
}

# Create folds for cross-validation
set.seed(123)
folds <- createFolds(data_cleaned$y, k = 5, list = TRUE, returnTrain = FALSE)

# Initialize results dataframe
results <- data.frame(
  Model = character(),
  Preprocessing = character(),
  Fold = integer(),
  MSE = double(),
  RMSE = double(),
  MAE = double(),
  R2 = double(),
  stringsAsFactors = FALSE
)

# List of models to test
models <- list(
  rf = function(X, y) {
    randomForest(
      x = X,
      y = y,
      ntree = 500,
      mtry = floor(ncol(X)/3),  # Default for regression is p/3
      nodesize = 5,
      importance = TRUE
    )
  },
  xgb = function(X, y) {
    xgboost(
      data = as.matrix(X),
      label = y,
      nrounds = 100,
      objective = "reg:squarederror",
      eta = 0.1,
      max_depth = 6,
      min_child_weight = 1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      verbose = 0
    )
  },
  lm = function(X, y) {
    lm(y ~ ., data = data.frame(X, y = y))
  }
)

# Preprocessing methods to test
preprocessing_methods <- c("standard", "pca")

# Main loop
for (preproc in preprocessing_methods) {
  cat("\nStarting preprocessing method:", preproc, "\n")
  
  for (fold_idx in 1:5) {
    cat("  Fold:", fold_idx, "\n")
    
    # Split data
    test_indices <- folds[[fold_idx]]
    train_data <- data_cleaned[-test_indices, ]
    test_data <- data_cleaned[test_indices, ]
    
    # Preprocess data
    use_pca <- preproc == "pca"
    train_processed <- preprocess_data(train_data, use_pca = use_pca)
    test_processed <- preprocess_data(test_data, use_pca = use_pca)
    
    # Store scaling parameters for y
    y_scaling <- attr(train_processed$y, "scaling")
    
    # For each model
    for (model_name in names(models)) {
      cat("    Model:", model_name, "\n")
      flush.console()
      
      # Train model
      model <- models[[model_name]](train_processed$X, train_processed$y)
      
      # Make predictions
      if (model_name == "xgb") {
        predictions <- predict(model, as.matrix(test_processed$X))
      } else {
        predictions <- predict(model, test_processed$X)
      }
      
      # Descale predictions and true values
      predictions_descaled <- descale_predictions(predictions, y_scaling)
      true_values_descaled <- descale_predictions(test_processed$y, y_scaling)
      
      # Evaluate
      metrics <- evaluate_model(predictions_descaled, true_values_descaled)
      
      # Store results
      results <- rbind(results, data.frame(
        Model = model_name,
        Preprocessing = preproc,
        Fold = fold_idx,
        MSE = metrics$MSE,
        RMSE = metrics$RMSE,
        MAE = metrics$MAE,
        R2 = metrics$R2
      ))
    }
  }
}

# Calculate summary statistics
summary_results <- aggregate(
  cbind(MSE, RMSE, MAE, R2) ~ Model + Preprocessing,
  data = results,
  FUN = function(x) c(mean = mean(x), sd = sd(x))
)

# Print results
cat("\nFinal Results:\n")
print(summary_results)

# Plot results
ggplot(results, aes(x = Model, y = MSE, fill = Preprocessing)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Model Performance Comparison",
       y = "Mean Squared Error",
       x = "Model")