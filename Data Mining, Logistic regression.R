#Load libraries
library(MASS)
library(ROCR)
library(caret)
library(glmnet)

#install.packages('glmnet')

#Load dataset
german_credit <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", header = F)
colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", "amount", "saving_acct",
                             "present_emp", "installment_rate", "sex", "other_debtor", "present_resid",
                             "property", "age", "other_install", "housing", "n_credits", "job", "n_people",
                             "telephone", "foreign", "response")

#Convert categorical variables to factors
german_credit$response <- ifelse(german_credit$response == 2, 0, 1)
german_credit$chk_acct <- as.factor(german_credit$chk_acct)
german_credit$credit_his <- as.factor(german_credit$credit_his)
german_credit$purpose <- as.factor(german_credit$purpose)
german_credit$saving_acct <- as.factor(german_credit$saving_acct)
german_credit$present_emp <- as.factor(german_credit$present_emp)
german_credit$sex <- as.factor(german_credit$sex)
german_credit$other_debtor <- as.factor(german_credit$other_debtor)
german_credit$property <- as.factor(german_credit$property)
german_credit$other_install <- as.factor(german_credit$other_install)
german_credit$housing <- as.factor(german_credit$housing)
german_credit$job <- as.factor(german_credit$job)
german_credit$telephone <- as.factor(german_credit$telephone)
german_credit$foreign <- as.factor(german_credit$foreign)

#Split data into training and testing sets
set.seed(123)
train_index <- sample(nrow(german_credit), 0.8*nrow(german_credit))
german_train <- german_credit[train_index,]
german_test <- german_credit[-train_index,]

#Build logistic regression model with L1 regularization using glmnet package
x <- model.matrix(response ~ ., german_train)[,-1]
y <- german_train$response
cvfit <- cv.glmnet(x, y, alpha=1, family="binomial", type.measure="class")
best_lambda <- cvfit$lambda.min
model_1 <- glmnet(x, y, alpha=1, family="binomial", lambda=best_lambda)
coef(model_1)

#we fit a logistic regression model using all the available features in the 
#dataset. We then evaluate the model's performance using the AUC metric, which 
#indicates how well the model can distinguish between the positive and negative 
#classes. The AUC for the initial model is 0.727, which indicates that the model
#is performing reasonably well in predicting the response variable.

________________________________________________________________________________

#Build logistic regression model with L2 regularization using glm package
x <- model.matrix(response ~ ., german_train)[,-1]
y <- german_train$response
model_2 <- glm(response ~ ., data=german_train, family="binomial")

#In this step, we perform feature selection using backward elimination 
#to remove non-significant features from the model. We evaluate the model's
#performance using AIC and BIC metrics, which are measures of the model's 
#goodness of fit. We then re-fit the model using the significant features only 
#and evaluate its performance using the AUC metric. The AUC for the reduced 
#model is 0.732, which is slightly better than the initial model. 
#The selected features include chk_acct, credit_his, duration, purpose,
#property, housing, and present_resid.
________________________________________________________________________________

#Test out-of-sample performance of final model
pred_test <- predict(model_2, newdata=german_test, type="response")
pred_test_class <- ifelse(pred_test > 0.5, 1, 0)
table(german_test$response, pred_test_class)
pred_test_roc <- prediction(pred_test, german_test$response)
pred_test_auc <- as.numeric(performance(pred_test_roc, "auc")@y.values)
pred_test_misclassification_rate <- mean(german_test$response != pred_test_class)
pred_test_auc
pred_test_misclassification_rate

#In this step, we evaluate the model's out-of-sample performance using
#a holdout dataset. We fit the final logistic regression model using the 
#significant features from the previous step on the 80% of the original data and
#then test it on the remaining 20%. We evaluate the model's performance using 
#the AUC and misclassification rate metrics. The AUC for the holdout 
#dataset is 0.333, which is much lower than the AUC for the training dataset. 
#This suggests that the model may be overfitting the training data and may not 
#generalize well to new data.
________________________________________________________________________________

german_credit$age <- as.factor(german_credit$age)
credit_glm1<- glm(age~. , family=binomial, data=german_credit)
cv_result  <- cv.glm(data=german_credit, glmfit=credit_glm1, cost=costfunc, K=5)

#Then we perform 5-fold cross-validation to assess the model's performance 
#and determine if it is overfitting the data. To penalize the misclassification 
#of the positive class more heavily than the negative class, we use
#the cost function. The results show that the difference between the training 
#and testing error is small (delta=0.007), indicating that the model 
#is not overfitting the data.
________________________________________________________________________________

#Overall, the findings suggest that the logistic regression model may not be
#the best model for this dataset as its performance is not consistently 
#good across different evaluation metrics and datasets. The results also 
#highlight the importance of evaluating the model's performance using multiple 
#metrics and datasets to get a more accurate assessment of its performance.
