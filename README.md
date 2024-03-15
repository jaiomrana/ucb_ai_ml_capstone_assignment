## Predictive Models for Credit Card Transaction Fraud Detection

**Jai Om**

### Executive Summary

**Project Overview and Goals:**

The goal of this project is to create a machine learning predictive model that can detect the credit card transactions fraud. Credit card frauds are common. It happens for multiple reasons such as due to stolen identity, lost credit card or when hackers hack the company's data. It's important to identify the fraud so that consumers aren't charged for the transaction(s) that they did not execute. It not only affects the consumers whose credit card was involved in the fraud but also costs millions to credit card company as most card companies provide protection to cardholders against frauds.

Adding fraud detection mechanism to card transactions might bring some inconvenience to cardholders, especially for False Positive cases, as they would have to confirm if they recognized the transaction (tagged as potential fraud) before transaction is approved by the card company. But in the end, avoiding the fraud is of greater importance than a minor inconvenience of confirming transaction's validity.

**Results and Conclusion**

Since the goal was to detect fraud, the Precision-Recall performance metrics was used to compare various models. Multiple classifier models were created and the best model provided the performance score of **~90%**. There were other classifier models that were not too far from the best. The models were tested with default threshold of 0.50 (i.e. 50% probability of fraud).

**Recommendation to the Card Companies**

Predictive models can be highly effective but there's also room for configuration. One such parameter that can be controlled by a Card Company is decision boundary threshold parameter. Current models were tested with the default value of 0.50 (i.e. 50% probability of fraud).

When a threshold is **0.90** - Such a high threshold exudes confidence in predictions and thereby missing out on some fraudulent transactions. Such a scenario is not desirable despite high Precision. Can you think of why?
It is because the business ends up paying a higher cost of missing out on fraud identification which is the sole purpose of building such a model. Note that the cost of a fraudulent transaction is much higher than the cost involved in blocked but legit transactions. 

When a threshold is **0.40** - Such a liberal threshold would block the majority of the transactions, which can annoy many customers. Not to forget the additional burden on human resources to work through the flagged transactions and identify the true frauds.

Thus, whether a transaction is predicted as fraudulent or legit depends largely on the threshold value. Thus it must be carefully chosen by business to find the right balance.

#### Business Objective
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. Business would like to get a tool that could predict if a given credit card transaction is likely a fraud or legit.

Business may not mind a certain percentage of false positive identification (labeling a genuine transaction as fraud) but avoiding false negatives (labeling a fraud transaction as genuine) is extremely important espcially for large amount transactions.

#### Research Question
The main question that this project aims to answer is - Can there be a good predictive model that can predict the credit card fraud to stop it from happening?

#### Data Sources
I have used the dataset from Kaggle which can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

As per details given at Kaggle, the dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

This dataset does not contain actual feature names but rather are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Class' is the target variable and it takes value 1 in case of fraud and 0 otherwise.

#### Methodology
Tree based classifier models were used to predict the class of the transaction as fraud or legit. The data was split between training and test datasets. Multiple models were trained using the training datset and then validated against the test dataset using cross-validation methods. The models are finally tuned to improve the performance using Grid Search techniques (e.g. GridSearchCV, RandomizedSearchCV).

Five models were trained, fine-tuned, and compared to find the best model for this task.
- **Logistic Regression** - This model provided the Precision-Recall score of **57.25%**
- **Random Forest Classifier** - This model provided the Precision-Recall score of **86.78%**
- **XGBoost** - This model provided the Precision-Recall score of **89.27%**
- **CatBoost** - This model provided the Precision-Recall score of **88.89%**
- **LightGBM** - This model provided the Precision-Recall score of **86.72%**

#### Results
I analyzed the card transactions data, checked for data imbalance, visualized the features to understand the relationship between the various features. Since the goal was to detect fraud, the Precision-Recall performance metrics was used to compare various models. Multiple classifier models were created and the best model provided the performance score of **~90%**. There were other classifier models that were not too far from the best. The models were tested with default threshold of 0.50 (i.e. 50% probability of fraud).

#### Next Steps
- The models were tuned with limited number of hyperparameters due to computational restrictions. More options can be tried to see if models can be improved further.
- Since the dataset is highly imbalanced, more sophisticated techniques can be applied to handle this, such as Synthetic Minority Over-sampling Technique (SMOTE) or Adaptive Synthetic Sampling (ADASYN). These can help improve model performance by balancing the dataset.
- Tuning existing models further or ensembling different models could futher enhance performance.

#### Outline of Project
[Link to download data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3)

[Link to notebook](https://github.com/jaiomrana/ucb_ai_ml_capstone_assignment/blob/main/credit_card_fraud_detection_models.ipynb)

### Contact and Further Information
Jai Om

Email: jaiomrana@yahoo.com
