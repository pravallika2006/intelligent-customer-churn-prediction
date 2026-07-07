# Business Understanding

## Project Overview
    The objective of this project is to develop an Intelligent Customer Churn Prediction and Decision Support System that predicts customers who are likely to leave a company's services, explains the reasons behind the prediction, categorizes customers based on risk, and suggests business actions to improve customer retention. 
## What is Customer Churn?
    when a customer decides to leave or stop using a company's service.

    What happens if 10,000 customers leave?

    Not only does revenue decrease, but:

    Marketing costs increase.
    Customer acquisition costs increase.
    Brand loyalty decreases.
    Company growth slows.
## Why is Customer Churn a Business Problem?
    Customer in general plays a huge role in a bussiness world and loosing them directly impacts the overall income,profit

## Why Use Machine Learning?
    A customer decides to leave a company not because of a single reason but many factors might influence his decision.So, to undertstand the dependency of his decision we use ML 
    Machine learning can learn these complex relationships from historical data and generalize them to new customers.
## Business Objective
    To identify customers who are likely to churn before they leave, allowing the business to take proactive actions that improve customer retention and reduce revenue loss.

## Stakeholders
    Any group or individual who effects or effected by a business decison are called stateholder
    Ex: customer support, ML Engineer, Data Scientist,Developer etc...

        | Stakeholder        | Responsibility                    |
    | ------------------ | --------------------------------- |
    | Business Manager   | Reduce churn and increase revenue |
    | Marketing Team     | Run retention campaigns           |
    | Customer Support   | Contact high-risk customers       |
    | Data Scientist     | Analyze data and build models     |
    | ML Engineer        | Deploy and maintain the model     |
    | Company Leadership | Make strategic business decisions |


## Success Criteria
    making sure the old customers stick to our services 
    in measurable terms:

    Lower churn rate.
    Higher customer retention.
    Better model performance.
    More effective marketing campaigns.
    Increased revenue.

    Business teams like metrics because they show whether the solution is working.
## Types of Prediction Errors
    1.False Positive:
    model might say the customer might leave ,but instaed he stays . this leads to uneccesaary expenditure(like discount offers) of safe customers 
    2.False Negatives:
    entirely opposite to False positive
    model might say they will stay but in real they'll leave 
    No focus on actual churn customers and loose of them 
## Why is this a Classification Problem?
    This project is a Binary Classification problem because the target variable has only two possible outcomes: **Churn** or **No Churn**. The Machine Learning model learns from historical customer data and classifies each customer into one of these two categories based on their features. 
## Key Takeaways
