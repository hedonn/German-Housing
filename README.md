German Housing Data Analysis
This repository contains code for analyzing the German Housing dataset using logistic regression. The goal of this project is to build a model that can predict whether a person will default on a housing loan based on their credit history and other features.


Getting Started
To get started with this project, you'll need to download the German Housing dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29). Once you have downloaded the dataset, you can run the german_housing_analysis.R file to analyze the data.


Prerequisites
To run the code in this project, you will need to have R and the following R packages installed:


MASS
caret
ROCR
glmnet


Running the Analysis
To run the analysis, simply run the german_housing_analysis.R file in R. The code will load the data, perform feature selection using backward elimination, fit a logistic regression model, and evaluate the model's performance using various metrics.

Results
The results of the analysis suggest that a logistic regression model may not be the best model for this dataset as its performance is not consistently good across different evaluation metrics and datasets. The AUC for the final model is 0.732, which is only slightly better than the AUC for the initial model (0.727). The model's out-of-sample performance is also poor, with an AUC of only 0.333 on the holdout dataset.

Conclusion
In conclusion, while a logistic regression model may not be the best model for this dataset, the analysis provides valuable insights into the factors that are most predictive of default on a housing loan. Future work could explore other machine learning models, such as random forests or neural networks, to improve the model's performance.
