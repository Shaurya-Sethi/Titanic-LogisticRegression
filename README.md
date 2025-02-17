---
Title: "Titanic Survival Prediction Using Logistic Regression"
Author: "Shaurya Sethi"
Date: "25th January 2025"
---

# Titanic Survival Prediction Using Logistic Regression in R

## Overview
This project uses **Logistic Regression** with **LASSO regularization** in R to predict whether a passenger onboard the Titanic survived or not. The dataset used is the famous **Titanic dataset**, which contains information about passengers such as age, gender, class, and survival status.

The goal of this project is to:
1. Perform exploratory data analysis (EDA) to understand the dataset.
2. Preprocess the data by handling missing values and encoding categorical variables.
3. Build a logistic regression model with LASSO regularization to predict survival.
4. Evaluate the model's performance using cross-validation, ROC curves, and confusion matrices.
5. Derive insights from the data and visualize key findings.

---

## Dataset
The dataset used in this project is the **Titanic dataset**, which is publicly available and often used for binary classification tasks. It includes the following features:
- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Binary indicator (0 = Did not survive, 1 = Survived).
- **Pclass**: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Name**: Passenger's name.
- **Sex**: Passenger's gender (male/female).
- **Age**: Passenger's age.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Missing Values**: Visualized using the `missmap` function from the `Amelia` package. Age had missing values, which were imputed based on passenger class. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/MissMap.png](#)
- **Survival Count**: Visualized the distribution of survival using a bar plot. The majority of passengers did not survive. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/SurvivalCount.png](#)
- **Passenger Classes**: Visualized the distribution of passenger classes. Most passengers were in the third class. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/PClasses.png](#)
- **Age Distribution**: Visualized the age distribution using a histogram. Most passengers were between 20 and 40 years old. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/AgeDist.png](#)
- **Gender Distribution**: Visualized the gender distribution. There were more male passengers than female passengers. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/GenderDist.png](#)

### 2. Data Preprocessing
- **Handling Missing Values**: Missing age values were imputed based on the median age of each passenger class. Missing titles were inferred based on gender and age. 
- **Feature Engineering**: Created a `FamilySize` feature by combining `SibSp` and `Parch`. Extracted titles (e.g., Mr, Mrs, Miss) from passenger names. 
- **Encoding Categorical Variables**: Converted categorical variables like `Pclass`, `Sex`, and `Embarked` into factors. 

### 3. Model Building
- **LASSO Logistic Regression**: Used LASSO regularization to perform feature selection and prevent overfitting. The optimal lambda value was determined using cross-validation. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/FeatureImportancePlot.png](#)
- **Cross-Validation**: Performed 10-fold cross-validation to evaluate the model's performance. 
- **Feature Importance**: Visualized the importance of features using a bar plot. 

### 4. Model Evaluation
- **Confusion Matrix**: Evaluated the model's performance using a confusion matrix. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/ConfMatrix.png](#)
- **ROC Curve**: Plotted the ROC curve and calculated the AUC (Area Under the Curve) to assess the model's predictive power. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/ROC.png](#)
- **Optimized Threshold**: Used Youden's Index to determine the best threshold for classification, improving model performance.

---

## Results

### Key Insights
1. **Socioeconomic Status and Survival**: First-class passengers had a higher survival rate compared to second and third-class passengers. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/SurvByPClass.png](#)
2. **Gender Disparity**: Female passengers had a significantly higher survival rate than male passengers. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/SurvByGender.png](#)
3. **Age and Survival**: Children and older passengers had higher survival rates compared to young adults. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/AgeVsSurv.png](#)
4. **Family Size and Survival**: Passengers with smaller family sizes had a higher chance of survival. [https://github.com/Shaurya-Sethi/Titanic-LogisticRegression/blob/main/SurvPropFamily.png](#)
5. **Fare and Survival**: Passengers who paid higher fares had a higher survival rate. 

### Model Performance
- **Cross-Validation Accuracy**: The model achieved an accuracy of approximately 80% on the training data. 
- **Optimized Threshold**: Using Youden's Index, the model's accuracy improved to 82%, with a precision of 88% and recall of 82%. 
- **AUC**: The AUC value was 0.86, indicating strong predictive power. 

### Performance Comparison Table
Below is a table comparing model performance metrics before and after threshold optimization:

| Metric       | Original Model | Optimized Threshold |
|--------------|----------------|---------------------|
| Accuracy     | 80.9%          | 82.6%               |
| Precision    | 82.8%          | 88.3%               |
| Recall       | 87.3%          | 82.7%               |
| F1-Score     | 85.0%          | 85.4%               |

---

### Visualizations
- **Survival Count**: Bar plot showing the distribution of survival. 
- **Passenger Classes**: Bar plot showing the distribution of passenger classes. 
- **Age Distribution**: Histogram showing the distribution of ages. 
- **Gender Distribution**: Bar plot showing the distribution of genders. 
- **Feature Importance**: Bar plot showing the importance of features selected by LASSO. 
- **ROC Curve**: Plot showing the ROC curve and AUC value. 
- **Confusion Matrix**: Heatmap showing the confusion matrix for the optimized model. 

---

## How to Run the Script

### 1. Install Dependencies
Ensure the following R packages are installed:
```R
install.packages(c("data.table", "Amelia", "ggplot2", "dplyr", "glmnet", "caret", "pROC", "caTools"))
```

### 2. Execute the Script
1. Clone this repository.
2. Set the working directory to the project folder in R.
3. Ensure the dataset files (`Titanic_train.csv` and `Titanic_test.csv`) are in the working directory.
4. Run the script to generate visualizations, preprocess the data, and build the model.

---

## Conclusion
This project demonstrates the use of logistic regression with LASSO regularization to predict survival on the Titanic. By performing thorough EDA, preprocessing, and model evaluation, we achieved a robust model with strong predictive power. The insights derived from the data provide valuable information about the factors influencing survival.

---

## Future Work
- Experiment with other machine learning algorithms (e.g., Random Forest, Gradient Boosting). 
- Perform hyperparameter tuning to further improve model performance. 
- Explore additional feature engineering techniques to enhance predictive power. 

---

## Author
**Shaurya Sethi**

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
