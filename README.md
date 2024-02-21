## Predictive Models for Credit Card Transaction Fraud Detection

**Jai Om**

### Executive summary

**Project overview and goals:**
The goal of this project is to create a machine learning predictive model that can detect the credit card transactions fraud. Credit card frauds are common. It happens for multiple reasons like due to stolen identity, lost credit card or when hackers hack the company's data. It's important to identify the fraud so that consumers aren't charged for the transaction(s) that they did not execute. It not only affects the consumers whose credit card was involved in the fraud but also costs millions to credit card company as most card companies provide protection to cardholders against frauds.

Adding fraud detection mechanism to card transactions might bring some inconvenience to cardholders, especially for False Positive cases, as they would have to confirm if they recognized the transaction (tagged as potential fraud) before transaction is approved by the card company. But in the end, avoiding the fraud is of greater importance than a minor inconvenience of confirming transaction's validity.

**Results and Conclusion**
I analyzed the data, checking for data unbalancing, visualizing the features and understanding the relationship between the various features. The best model, so far, is a classifier model that provided the performance score of **~99%**. The other classifier model provided a lower performance score of **~89%**.

#### Business Objective
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. Business would like to get a tool that could predict if a given credit card transaction is likely a fraud or legit.

Business may not mind a certain percentage of false positive identification (labeling a genuine transaction as fraud) but avoiding false negatives (labeling a fraud transaction as genuine) is extremely important espcially for large amount transactions.

#### Research Question
Can the credit card fraud be predicted before authorization to stop it from happening?

#### Data Sources
I have used the dataset from Kaggle which can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

As per details given at Kaggle, the dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

This dataset does not contain actual feature names but rather are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Class' is the target variable and it takes value 1 in case of fraud and 0 otherwise.

#### Methodology
I have used the tree based classifier model to predict the class of the transaction as fraud or legit. The data was split between training and test datasets. Multiple models were trained using the training datset and then validated against the test dataset using cross-validation methods. The models are finally tuned to improve the performance using Grid Search techniques (e.g. GridSearchCV, RandomizedSearchCV).

#### Results
The best model, so far, is based on XGBoost classifier algorithm with **0.98671104 ROC-AUC** score (Area under the ROC Curve). The RandomForestClassifier model provided ROC-AUC score of 0.88263548. These model will further be tuned to improve the performance.

#### Next steps
I'll add some more classifier models to see if a better model can be created using other machine learning algorithms. I'll also continue to work on the models to further tune them to improve the performance.

#### Outline of project
[Link to download data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3)

[Link to notebook](https://github.com/jaiomrana/ucb_ai_ml_capstone_assignment/blob/main/credit_card_fraud_detection_models.ipynb)

### Contact and Further Information
Jai Om

Email: jaiomrana@yahoo.com
