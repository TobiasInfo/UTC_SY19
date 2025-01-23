# Machine Learning Approaches for Classification and Regression

## Project Overview
This project focuses on applying statistical and machine learning methods to address classification and regression problems. The analysis is performed on both real and simulated datasets, emphasizing data exploration, preprocessing, and the evaluation of predictive models. The project demonstrates key methodologies and insights relevant to machine learning applications in real-world scenarios, particularly in the health insurance domain.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Data Transformation and Encoding](#data-transformation-and-encoding)
   - [Handling Outliers](#handling-outliers)
3. [Machine Learning Models](#machine-learning-models)
   - [Classification](#classification)
   - [Regression](#regression)
4. [Results and Insights](#results-and-insights)
5. [Limitations and Future Work](#limitations-and-future-work)
6. [Conclusion](#conclusion)

---

## Introduction
This project investigates factors influencing medical insurance costs using a dataset obtained from Kaggle. It aims to answer questions such as:
- What are the key variables affecting medical costs?
- How can regional differences and demographic factors impact costs?
- How accurately can we classify individuals as smokers or non-smokers?

We also explore simulated datasets to test classification and regression techniques on synthetic data structures, highlighting the flexibility and robustness of machine learning approaches.

---

## Data Exploration and Preprocessing

### Exploratory Data Analysis (EDA)
- **Dataset Overview:** The real dataset includes 1,337 unique entries with variables such as age, gender, BMI, smoker status, and medical charges.
- **Key Findings:**
  - The average medical cost is \$13,279, with significant variability.
  - 20.5% of the individuals are smokers, a key factor driving higher costs.
  - BMI, age, and smoking status show strong correlations with medical charges.

### Data Transformation and Encoding
- **Categorical Variables:** Gender, smoker status, and region are encoded using label encoding and one-hot encoding.
- **Numerical Variables:** Variables such as BMI and age are normalized to improve model performance.

### Handling Outliers
Outliers in variables like BMI, age, and medical charges were identified and treated using interquartile range (IQR) methods to enhance model robustness.

---

## Machine Learning Models

### Classification
- **Objective:** Classify individuals as smokers or non-smokers.
- **Challenges:** The dataset has an imbalance, with only 20.5% of individuals being smokers.
- **Models Tested:** Logistic Regression, SVM (Gaussian kernel), Random Forest, Decision Trees.
- **Best Model:** Gaussian SVM with down-sampling achieved a perfect recall of 1.0 for the smoker class, ensuring all smokers were correctly classified.

### Regression
- **Objective:** Predict medical charges based on demographic and health factors.
- **Approach:**
  - Linear regression models (Ridge, LASSO) performed adequately.
  - Generalized Additive Models (GAM) with cubic splines outperformed others, capturing non-linear relationships between variables.

---

## Results and Insights
- Smoking status, BMI, and age were identified as the most significant factors influencing medical charges.
- For classification, the Gaussian SVM provided high recall for minority classes (smokers), minimizing critical financial errors.
- GAMs demonstrated flexibility in capturing complex relationships for regression tasks, outperforming linear models in predictive accuracy.

---

## Limitations and Future Work
- **Limitations:**
  - Imbalanced datasets posed challenges for classification.
  - Computational constraints limited the exploration of advanced architectures.
- **Future Work:**
  - Implement advanced sampling techniques to handle class imbalance.
  - Explore interaction terms and non-linear relationships in more depth.
  - Optimize hyperparameters for all models to maximize performance.

---

## Conclusion
This project highlights the importance of tailored machine learning approaches for different types of data and objectives. By leveraging Gaussian SVMs for classification and GAMs for regression, we achieved significant predictive accuracy while maintaining interpretability. These findings provide valuable insights into the application of machine learning in health insurance and beyond.

---

## Authors
- **Tobias Savary**
- **Nassim Saidi**

## Acknowledgments
This project was conducted as part of the SY19 Machine Learning module at the Université de Technologie de Compiègne, academic year 2024/2025.

---

## How to Run the Project
1. Clone the repository.
2. Install the required Python libraries listed in `requirements.txt`.
3. Run the Jupyter notebooks for:
   - Data exploration and preprocessing.
   - Model training and evaluation.

---

## License
This project is open-source and available under the MIT License.
