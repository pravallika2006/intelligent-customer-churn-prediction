- Why did you use SHAP?

    Machine learning models like Random Forest provide accurate predictions but are difficult to interpret. I integrated SHAP to explain individual predictions so users can understand which customer attributes increased or decreased churn risk. This makes the model more transparent and trustworthy.

- If a company can only contact 1,000 customers this month, but your model predicts 10,000 customers are likely to churn, how would you decide which 1,000 customers to prioritize?
    - i think classifying the 10k customers based on risk or who is more liskely to leave early and categorise them into high,medium,low risk
    - Instead of treating every predicted churn customer equally, we:
    Rank customers by their churn probability.
    Divide them into risk groups.
    Focus on the highest-risk customers first because the business has limited resources.

- How did you validate the quality of your dataset before preprocessing?
    - I checked for missing values using isnull(), searched for hidden missing values stored as blank strings, verified duplicate rows and duplicate customer IDs, inspected the number of unique values in each feature, and reviewed categorical values for inconsistencies before making any preprocessing decisions.

- What was the most important insight from your EDA?
    - One of the strongest findings was that contract duration had a major impact on churn. Customers on Month-to-Month contracts churned far more frequently than those on One-Year or Two-Year contracts. This suggested that encouraging long-term subscriptions could be an effective business strategy. I later verified the importance of this feature using feature importance and SHAP explanations.

- Why did Logistic Regression outperform XGBoost?
    - The Telco Churn dataset has strong linear relationships between several customer attributes and churn. Logistic Regression generalized better on the test data, while more complex tree-based models did not provide additional predictive benefit without hyperparameter tuning.
