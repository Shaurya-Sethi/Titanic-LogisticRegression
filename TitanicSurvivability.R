library(data.table)
library(Amelia)
library(ggplot2)
library(dplyr)
library(glmnet)


#loading datasets
df.train <- fread("Titanic_train.csv")
df.test <- fread("Titanic_test.csv")

#EDA
#####

missing.val.plot <- missmap(df.train, main = "Missing Map",
                            col = c("yellow","black"),
                            legend = FALSE)
print(missing.val.plot)
#Clearly age has missing values

#Visualising those who survived vs those who did not
pl <- ggplot(df.train, aes(x = factor(Survived), fill = factor(Survived))) +
        geom_bar() +
        geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
        scale_fill_manual(values = c("red", "green"), labels = c("Did not survive", "Survived")) +
        labs(
          title = "Titanic Survival Count",
          x = "Survival Status",
          y = "Number of People",
          fill = "Survival"
        ) +
        theme_minimal(base_size = 14) + # Cleaner theme
        theme(
          plot.title = element_text(hjust = 0.5), # Center align title
          legend.position = "top", # Place legend at the top
          axis.text.x = element_text(size = 12), # Adjust x-axis text size
          axis.text.y = element_text(size = 12) # Adjust y-axis text size
        )
print(pl) #Survival count is low compared to deceased

#Visualising Passenger Classes

pl2 <- ggplot(df.train, aes(Pclass)) +
  geom_bar(aes(fill = factor(Pclass)),color = "black", alpha = 0.5) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("gold", "grey","brown"), labels = c("First Class", "Second Class", "Third Class")) +
  labs(
    title = "Passenger Classes",
    x = "Class Type",
    y = "Number of People",
    fill = "Pclass"
  ) +
  theme_minimal(base_size = 14) + # Cleaner theme
  theme(
    plot.title = element_text(hjust = 0.5), # Center align title
    legend.position = "top", # Place legend at the top
    axis.text.x = element_text(size = 12), # Adjust x-axis text size
    axis.text.y = element_text(size = 12) # Adjust y-axis text size
  )
print(pl2) #Most people are in the third class


#Visualising Ages on board

pl3 <- ggplot(df.train, aes(Age)) +
  geom_histogram(
    bins = 20,
    binwidth = 5,
    color = "black",
    fill = "blue",
    alpha = 0.7
  ) +
  scale_x_continuous(breaks = seq(0, 80, by = 5)) + # Set custom x-axis breaks
  labs(
    title = "Distribution of Ages on Board",
    x = "Age",
    y = "Count"
  ) +
  theme_minimal(base_size = 14)
print(pl3) #Most people are aged between 20 and 40 but there are quite a few children onboard too.

#Males vs Females

pl4 <- ggplot(df.train, aes(x = factor(Sex), fill = factor(Sex))) +
    geom_bar(color = "black", alpha = 0.7) +
    geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
    scale_fill_manual(values = c("pink", "blue"), labels = c("Female", "Male")) +
    labs(
      title = "Distribution of Gender",
      x = "Gender",
      y = "Number of People",
      fill = "Gender"
    ) +
    theme_minimal(base_size = 14) + # Cleaner theme
    theme(
      plot.title = element_text(hjust = 0.5), # Center align title
      legend.position = "top", # Place legend at the top
      axis.text.x = element_text(size = 12), # Adjust x-axis text size
      axis.text.y = element_text(size = 12) # Adjust y-axis text size
    )
print(pl4) #More males than females

#Data Cleaning
#Handling missing values in Age (Visualised using missmap)


# Calculating median ages for each Pclass
median_ages <- df.train %>%
  group_by(Pclass) %>%
  summarize(median_age = median(Age, na.rm = TRUE))

# Plotting with boxplot and median annotations
pl5 <- ggplot(df.train, aes(x = factor(Pclass), y = Age)) +  # Treat Pclass as a factor (categorical)
  geom_boxplot(aes(group = Pclass, fill = factor(Pclass)), color = "black", alpha = 0.7) +
  geom_text(
    data = median_ages, 
    aes(x = factor(Pclass), y = median_age, label = round(median_age, 1)), 
    vjust = -0.5, 
    size = 5
  ) +
  scale_fill_manual(
    values = c("gold", "grey", "brown"), 
    labels = c("First Class", "Second Class", "Third Class")
  ) +
  labs(
    title = "Passenger Classes",
    x = "Class Type",
    y = "Age",
    fill = "Pclass"
  ) +
  theme_minimal(base_size = 14) + # Cleaner theme
  theme(
    plot.title = element_text(hjust = 0.5), # Center align title
    legend.position = "top", # Place legend at the top
    axis.text.x = element_text(size = 12), # Adjust x-axis text size
    axis.text.y = element_text(size = 12) # Adjust y-axis text size
  )
print(pl5) # 1st Class passengers are older on average than second who are in turn older than 3rd.

#Replacing missing age values with avg age depending on Pclass
impute_age <- function(age,class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- 37
        
      }else if (class[i] == 2){
        out[i] <- 29
        
      }else{
        out[i] <- 24
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}

fixed.ages <- impute_age(df.train$Age,df.train$Pclass)
df.train$Age <- fixed.ages

#check for na values again to test if the function successfully replaced them

print(missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE))


#Deriving Insights from the dataset

# Insight 1: Socioeconomic Status and Survival
# Visualizing survival rates across passenger classes (Pclass)
pl_insight1 <- ggplot(df.train, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black", alpha = 0.7) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Survival Proportion by Passenger Class",
    x = "Passenger Class",
    y = "Proportion Survived",
    fill = "Survived"
  ) +
  theme_minimal()
print(pl_insight1)

# Insight 2: Gender Disparity
# Visualizing survival rates by gender
pl_insight2 <- ggplot(df.train, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black", alpha = 0.7) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Survival Proportion by Gender",
    x = "Gender",
    y = "Proportion Survived",
    fill = "Survived"
  ) +
  theme_minimal()
print(pl_insight2)

# Insight 3: The Impact of Age
# Visualizing age distribution by survival status
pl_insight3 <- ggplot(df.train, aes(x = Age, fill = factor(Survived))) +
  geom_histogram(binwidth = 5, position = "identity", alpha = 0.7, color = "black") +
  facet_wrap(~ Survived) +
  labs(
    title = "Age Distribution by Survival Status",
    x = "Age",
    y = "Count"
  ) +
  theme_minimal()
print(pl_insight3)

# Insight 4: Family Size and Survival
# Creating a FamilySize variable
df.train$FamilySize <- df.train$SibSp + df.train$Parch + 1

# Visualizing survival rates by family size
pl_insight4 <- ggplot(df.train, aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black", alpha = 0.7) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Survival Proportion by Family Size",
    x = "Family Size",
    y = "Proportion Survived",
    fill = "Survived"
  ) +
  theme_minimal()
print(pl_insight4)

# Insight 5: Fare and Survival
# Visualizing survival rates by fare
pl_insight5 <- ggplot(df.train, aes(x = Fare, fill = factor(Survived))) +
  geom_histogram(binwidth = 10, position = "identity", alpha = 0.7, color = "black") +
  facet_wrap(~ Survived) +
  labs(
    title = "Fare Distribution by Survival Status",
    x = "Fare",
    y = "Count"
  ) +
  theme_minimal()
print(pl_insight5)



#######
# Building a Logistic Regression model


#feature engineering

# Extract Title from Name
df.train$Title <- sub(".*, (.*?)\\..*", "\\1", df.train$Name)

# Check the distribution of titles
print(table(df.train$Title))


# Insight 6: Title and Survival
# Visualizing survival rates by title
pl_insight6 <- ggplot(df.train, aes(x = Title, fill = factor(Survived))) +
  geom_bar(position = "fill", color = "black", alpha = 0.7) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Survival Proportion by Title",
    x = "Title",
    y = "Proportion Survived",
    fill = "Survived"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(pl_insight6)

#dropping irrelevant features
df.train <- df.train %>% select(-PassengerId, -Name, -Cabin, -Ticket)

#encoding categorical variables
df.train$Survived <- factor(df.train$Survived)
df.train$Pclass <- factor(df.train$Pclass)
df.train$Parch <- factor(df.train$Parch)
df.train$SibSp <- factor(df.train$SibSp)
df.train$FamilySize <- factor(df.train$FamilySize)
df.train$Embarked <- factor(df.train$Embarked)

# Prepare data for LASSO
x <- model.matrix(Survived ~ ., data = df.train)[, -1]  # Feature matrix
y <- as.numeric(as.character(df.train$Survived))  # Target vector

# Perform LASSO with cross-validation
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")

# Optimal lambda values
best_lambda <- lasso_cv$lambda.min
lambda_1se <- lasso_cv$lambda.1se

cat("Best Lambda (min):", best_lambda, "\n")
cat("Lambda within 1-SE:", lambda_1se, "\n")

# Plotting the cross-validation curve
plot(lasso_cv)

# Fit the final LASSO model
final_lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")

# Extract selected features
lasso_coefficients <- coef(final_lasso_model)
print(lasso_coefficients)

# Preparing data for feature importance plot
feature_importance <- data.frame(
  Feature = rownames(lasso_coefficients)[-1],  # Exclude intercept
  Importance = abs(lasso_coefficients[-1, 1]) # Absolute value of coefficients
)

# Sort features by importance
feature_importance <- feature_importance[order(-feature_importance$Importance), ]

# Plot feature importance

print(ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Feature Importance from LASSO Model",
    x = "Features",
    y = "Coefficient Magnitude"
  ) +
  theme_minimal())

# Extract coefficients as a matrix
lasso_coefficients <- as.matrix(coef(final_lasso_model))

# Extract features with non-zero coefficients
selected_features <- rownames(lasso_coefficients)[lasso_coefficients[, 1] != 0]  # Filter non-zero coefficients
selected_features <- selected_features[-1]  # Remove the intercept
print(selected_features)


# Preparing test data
# Ensuring categorical variables in the test data are factors

df.test$Title <- sub(".*, (.*?)\\..*", "\\1", df.test$Name)
df.test <- df.test %>% select(-PassengerId, -Name, -Cabin, -Ticket)
df.test$FamilySize <- df.test$SibSp + df.test$Parch + 1

df.test$Pclass <- factor(df.test$Pclass, levels = c(1, 2, 3))
df.test$Sex <- factor(df.test$Sex, levels = c("male", "female"))
df.test$Embarked <- factor(df.test$Embarked, levels = c("C", "Q", "S"))
df.test$Title <- factor(df.test$Title, levels = c("Don", "Jonkheer", "Master","Miss", "Mr", "Mrs", "Rev", "Sir"))
df.test$FamilySize <- factor(df.test$FamilySize)

print(missmap(df.test, main = "Missing Map",
              col = c("yellow","black"),
              legend = FALSE))
#Missing values in Age again

#Replacing missing age values with avg age depending on Pclass again
impute_age2 <- function(age,class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- 37
        
      }else if (class[i] == 2){
        out[i] <- 29
        
      }else{
        out[i] <- 24
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}

fixed.ages2 <- impute_age2(df.test$Age,df.test$Pclass)
df.test$Age <- fixed.ages2

#There are also 5 na values in titles (from missmap)
replace_na_titles <- function(data) {
  data$Title <- as.character(data$Title)  # Convert Title to character for replacement
  
  for (i in 1:nrow(data)) {
    if (is.na(data$Title[i])) {
      if (data$Sex[i] == "male") {
        if (data$Age[i] < 18) {
          data$Title[i] <- "Master"
        } else {
          data$Title[i] <- "Mr"
        }
      } else if (data$Sex[i] == "female") {
        if (data$Age[i] < 18) {
          data$Title[i] <- "Miss"
        } else {
          data$Title[i] <- "Mrs"
        }
      }
    }
  }
  
  data$Title <- factor(data$Title, levels = c("Don", "Jonkheer", "Master", "Miss", "Mr", "Mrs", "Rev", "Sir"))  # Re-encode Title as factor
  return(data)
}
df.test <- replace_na_titles(df.test)

# 1 NA value in Fare
df.test$Fare[is.na(df.test$Fare)] <- median(df.train$Fare, na.rm = TRUE)

# Create the model matrix for the test data
x_test <- model.matrix(~ ., data = df.test)[, -1]  # Remove intercept column

#Ensuring feature alignment between training and test data (Main Issue Faced in Project)
training_features <- colnames(x)
print(training_features)

missing_features <- setdiff(training_features, colnames(x_test))
print(missing_features)

extra_features <- setdiff(colnames(x_test), training_features)
print(extra_features)

# Adding missing features to x_test
missing_features <- setdiff(training_features, colnames(x_test))
for (feature in missing_features) {
  x_test <- cbind(x_test, setNames(data.frame(rep(0, nrow(x_test))), feature))
}

# Reorder columns to match training_features
x_test <- x_test[, training_features, drop = FALSE]

# Verify alignment
if (!all(colnames(x_test) == training_features)) {
  stop("Feature alignment failed!")
}

# Check dimensions
print(dim(x))       # Training data dimensions
print(dim(x_test))  # Test data dimensions

x_test <- as.matrix(x_test)

# Predicted probabilities and classes
predictions <- predict(final_lasso_model, newx = x_test, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Since we don't have ground truth labels available for our test test, we can't evaluate using confusion matrix
# We will thus use cross-validation instead



# Define the cross-validation control
train_control <- trainControl(method = "cv", number = 10)  # 10-fold CV

# Perform cross-validation with glmnet
cv_model <- train(
  Survived ~ .,
  data = df.train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 1, lambda = best_lambda)  # Use your best lambda from LASSO
)

# Print the results
print(cv_model)

# Extract performance metrics
cat("Cross-Validation Accuracy:", max(cv_model$results$Accuracy), "\n")

# Add predicted probabilities to the test data
df.test$Survival_Probability <- predict(final_lasso_model, newx = as.matrix(x_test), type = "response")

# Plot a histogram of predicted probabilities
print(ggplot(df.test, aes(x = Survival_Probability)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Predicted Survival Probabilities",
    x = "Predicted Probability of Survival",
    y = "Passenger Count"
  ) +
  theme_minimal())

# Sort by predicted probabilities
most_likely <- df.test[order(-df.test$Survival_Probability), ][1:5, ]
least_likely <- df.test[order(df.test$Survival_Probability), ][1:5, ]

# Display the top 5 most likely to survive
print("Top 5 Most Likely to Survive:")
print(most_likely[, c("Pclass", "Sex", "Age", "Fare", "Survival_Probability")])

# Display the top 5 least likely to survive
print("Top 5 Least Likely to Survive:")
print(least_likely[, c("Pclass", "Sex", "Age", "Fare", "Survival_Probability")])




### Alternatively splitting training data into test and training set also works for assessing model performance

library(pROC)
library(caTools)

# Split the data using caTools
set.seed(42)
split <- sample.split(df.train$Survived, SplitRatio = 0.8)  # 80% training, 20% testing
train_data <- subset(df.train, split == TRUE)
test_data <- subset(df.train, split == FALSE)

# Prepare data for LASSO (training set)
x_train2 <- model.matrix(Survived ~ ., data = train_data)[, -1]
y_train2 <- as.numeric(as.character(train_data$Survived))

# Perform LASSO logistic regression with cross-validation
lasso_cv2 <- cv.glmnet(x_train2, y_train2, alpha = 1, family = "binomial")
best_lambda <- lasso_cv2$lambda.min

# Fit the final LASSO model
final_lasso_model2 <- glmnet(x_train2, y_train2, alpha = 1, lambda = best_lambda, family = "binomial")

# Prepare test data
x_test2 <- model.matrix(Survived ~ ., data = test_data)[, -1]

# Ensure feature alignment between training and test data
training_features <- colnames(x_train2)
test_features <- colnames(x_test2)

# Find missing and extra features
missing_features <- setdiff(training_features, test_features)
extra_features <- setdiff(test_features, training_features)

# Add missing features to x_test2 with default values of 0
for (feature in missing_features) {
  x_test2 <- cbind(x_test2, setNames(data.frame(rep(0, nrow(x_test2))), feature))
}

# Remove extra features from x_test2
x_test2 <- x_test2[, training_features, drop = FALSE]

# Verify alignment
if (!all(colnames(x_test2) == training_features)) {
  stop("Feature alignment between training and test data failed!")
}

# Convert to matrix
x_test2 <- as.matrix(x_test2)
y_test2 <- as.numeric(as.character(test_data$Survived))

# Predict on the test set
test_predictions <- as.numeric(predict(final_lasso_model2, newx = x_test2, type = "response"))

# Evaluate performance using the default threshold (0.5)
test_pred_classes <- ifelse(test_predictions > 0.5, 1, 0)
conf_matrix <- confusionMatrix(factor(test_pred_classes), factor(y_test2))
print(conf_matrix)

# Plot ROC curve
roc_curve <- roc(y_test2, test_predictions)
plot(roc_curve, col = "blue", main = "ROC Curve for LASSO Logistic Regression")
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

# Find the best threshold using Youden's Index
best_threshold <- coords(roc_curve, "best", ret = "threshold", best.method = "youden")

# Extract the threshold value from the data frame or list
if (is.list(best_threshold)) {
  best_threshold_value <- best_threshold[[1]]  # Extract the value if it's a list
} else {
  best_threshold_value <- as.numeric(best_threshold)  # Handle direct numeric value
}

cat("Best Threshold (Youden's Index):", best_threshold_value, "\n")

# Predict using the best threshold
test_pred_classes_optimized <- ifelse(test_predictions > best_threshold_value, 1, 0)

# Recompute confusion matrix and metrics using the new threshold
conf_matrix_optimized <- confusionMatrix(factor(test_pred_classes_optimized), factor(y_test2))
print(conf_matrix_optimized)

# Extract key metrics using the optimized threshold
accuracy_optimized <- conf_matrix_optimized$overall["Accuracy"]
precision_optimized <- conf_matrix_optimized$byClass["Precision"]
recall_optimized <- conf_matrix_optimized$byClass["Recall"]
f1_score_optimized <- 2 * ((precision_optimized * recall_optimized) / (precision_optimized + recall_optimized))

cat("Optimized Accuracy:", accuracy_optimized, "\n")
cat("Optimized Precision:", precision_optimized, "\n")
cat("Optimized Recall:", recall_optimized, "\n")
cat("Optimized F1-Score:", f1_score_optimized, "\n")

#Comparing previous vs improved model

library(knitr)

# Compare Performance Metrics: Original vs. Improved
performance_comparison <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score"),
  Original = c(
    conf_matrix$overall["Accuracy"],
    conf_matrix$byClass["Precision"],
    conf_matrix$byClass["Recall"],
    2 * ((conf_matrix$byClass["Precision"] * conf_matrix$byClass["Recall"]) / 
           (conf_matrix$byClass["Precision"] + conf_matrix$byClass["Recall"]))
  ),
  Improved = c(
    accuracy_optimized,
    precision_optimized,
    recall_optimized,
    f1_score_optimized
  )
)

# Round the metrics for cleaner presentation
performance_comparison$Original <- round(performance_comparison$Original, 3)
performance_comparison$Improved <- round(performance_comparison$Improved, 3)

# Create and display the comparison table
print(kable(
  performance_comparison,
  format = "markdown",
  caption = "Comparison of Model Performance: Original vs. Improved Threshold"
))

library(caret)
library(ggplot2)

# Generate a confusion matrix
conf_matrix_df <- as.data.frame(conf_matrix_optimized$table)
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")

# Plot the confusion matrix as a heatmap
print(ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(
    title = "Confusion Matrix",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal())

# Create a data frame with predicted and actual values
comparison_df <- data.frame(
  Category = c(rep("Predicted", length(test_pred_classes_optimized)), 
               rep("Actual", length(y_test2))),
  Survived = c(test_pred_classes_optimized, y_test2)
)

# Plot bar chart of predicted vs actual class distribution
print(ggplot(comparison_df, aes(x = factor(Survived), fill = Category)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("blue", "green"), labels = c("Actual", "Predicted")) +
  labs(
    title = "Comparison of Predicted and Actual Class Distribution",
    x = "Survival (0 = Did Not Survive, 1 = Survived)",
    y = "Count",
    fill = "Category"
  ) +
  theme_minimal())

# Add predicted probabilities to the test data
test_data$Survival_Probability <- test_predictions

# Plot predicted probabilities grouped by actual survival status
print(ggplot(test_data, aes(x = factor(Survived), y = Survival_Probability, fill = factor(Survived))) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  scale_fill_manual(values = c("red", "green"), labels = c("Did Not Survive", "Survived")) +
  labs(
    title = "Predicted Probabilities by Actual Survival Status",
    x = "Actual Survival",
    y = "Predicted Probability of Survival",
    fill = "Actual Survival"
  ) +
  theme_minimal())








