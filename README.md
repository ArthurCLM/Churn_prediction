![](Images/ChurnProject.png)

# 1. Churn_prediction
This represents a end-to-end machine learning project employing XGBoost for forecasting the likelihood of customer churn within a credit card service offered by a bank. The detection of potential churners aids in devising retention strategies, ensuring a sustainable revenue stream.

# 2. Business problem and objectives
We aim to accomplist the following for this study:

- Identify and visualize which factors contribute to customer churn:

- Build a prediction model that will perform the following:
  - Classify if a customer is going to churn or not.
  - Preferably and based on model performance, choose a model that will attach a probability to the churn to make it easier for customer service to target low hanging fruits in their efforts to prevent churn

# 3. Technologies and tools
The technologies and tools used were Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, Category-Encoders, Scikit-Optimize, Xgboost), Jupyter Notebook, Git and Github (version control), machine learning classification algorithms, statistics, and Google Colab.

# 4. Business Insights

1. Approximately 20% of the customers have experienced churn. Consequently, a plausible baseline model might involve predicting that 20% of customers will churn. Given that 20% represents a relatively small portion, it becomes imperative to ensure that the selected model excels at accurately identifying this 20%, as it is of significant interest to the bank to pinpoint and retain this group, even at the potential expense of accurately predicting the customers who will remain with the bank.
   
![](Images/Prop_Churn.png)

2. Although the majority of the data pertains to individuals from France, there exists an inverse relationship between the proportion of churned customers and the customer population in various regions. This suggests a potential issue within the bank, possibly linked to insufficient customer service resources in areas with fewer clients.

3. The percentage of female customers experiencing churn is notably higher than that of male customers.

4. Interestingly, a significant portion of the customers who churned held credit cards. However, given that the majority of customers have credit cards, this correlation could be coincidental.

5. As expected, inactive members exhibit a higher churn rate. Concerningly, the overall proportion of inactive members is quite substantial, indicating a potential need for implementing a program aimed at converting this group into active customers. Such an initiative could significantly reduce customer churn.

![](Images/Count_feature_churn.png)

6. There is no notable disparity in the distribution of credit scores between customers who have been retained and those who have churned.

7. Older customers are churning at a higher rate than their younger counterparts, indicating potential differences in service preferences across age categories. This suggests that the bank may need to reassess its target market or revise its retention strategies for distinct age groups.

8. In terms of tenure, customers at both ends of the spectrum (those with minimal and extensive tenure) are more prone to churn compared to those with an average tenure.

9. Alarmingly, the bank is experiencing customer attrition among those with substantial account balances, which could impact the bank's available capital for lending.

10. Neither the choice of product nor salary level appears to significantly influence the likelihood of churn.

![](Images/Boxplot_feature_churn.png)

# 5. Modelling

# 6. Financial results

# 7. How to apply the model in your machine

# 8. Dataset link






