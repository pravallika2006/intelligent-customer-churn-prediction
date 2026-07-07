# Exploratory Data Analysis

## Target Variable Analysis

### Objective
Understand the distribution of churned and non-churned customers.

### Observations
- Total customers: 7043
- Customers retained: 5174 (73.46%)
- Customers churned: 1869 (26.54%)
- Approximately one out of every four customers has churned.

### Business Insight
The dataset shows a moderate class imbalance, with significantly more retained customers than churned customers. Predicting churn accurately can help the business proactively retain customers through targeted offers, improved support, or personalized retention strategies.

## Feature Analysis: Tenure

### Objective
Analyze whether customer tenure influences churn.

### Observations
- Average tenure is 32.37 months.
- Customers who stayed have a much higher median tenure than customers who churned.
- The tenure distribution is bimodal, with many new customers and many long-term customers.
- Most churned customers have relatively low tenure.

### Business Insight
Customers are most vulnerable to churn during their early months with the company. Improving the onboarding experience and engaging new customers could reduce churn significantly.

### ML Insight
Tenure appears to be a highly informative feature and is expected to contribute strongly to churn prediction.

## Feature Analysis: MonthlyCharges

### Objective
Analyze whether monthly charges influence customer churn.

### Observations
- Monthly charges range from 18.25 to 118.75.
- Customers who churn have a higher median monthly charge than customers who stay.
- The distribution suggests multiple pricing plans offered by the company.

### Business Insight
Customers paying higher monthly charges are more likely to churn. Premium customers may have higher expectations regarding service quality and customer support. Improving the experience for these customers could reduce churn.

### ML Insight
MonthlyCharges appears to be an informative feature for churn prediction because its distribution differs between churned and retained customers.

## Feature Analysis: Contract

### Objective
Analyze whether contract duration influences customer churn.

### Observations
- Around 55% of customers are on Month-to-Month contracts.
- Month-to-Month customers have the highest churn rate.
- Two-Year contract customers have the lowest churn rate.
- Churn decreases as contract duration increases.

### Business Insight
Customers with long-term contracts are more loyal because they are committed for a longer period. Encouraging Month-to-Month customers to switch to yearly or two-year plans through discounts, rewards, or exclusive benefits could significantly reduce churn.

### ML Insight
Contract is one of the strongest predictors of churn. The contract type provides valuable information for estimating a customer's likelihood of leaving.

## Feature Analysis: InternetService

### Objective
Analyze whether the type of internet service influences customer churn.

### Observations
- Fiber Optic is the most commonly used internet service.
- Fiber Optic customers have the highest churn rate (~42%).
- DSL customers have a moderate churn rate (~19%).
- Customers with no internet service have the lowest churn rate (~7%).

### Business Insight
Although Fiber Optic is a premium service, customers using it churn more frequently. This may be due to higher pricing, stronger competition, greater customer expectations, or service quality issues. The company should investigate customer feedback and improve the Fiber Optic experience.

### ML Insight
InternetService appears to be an important categorical feature because churn rates differ significantly across service types, making it useful for predictive modeling.

### Online Security

- Customers without Online Security have the highest churn rate (~42%).
- Customers with Online Security have a much lower churn rate (~15%).
- Customers without internet service show the lowest churn (~7%).
- Online Security appears to be one of the strongest indicators of customer retention.
- This feature is expected to be highly informative for the machine learning model.