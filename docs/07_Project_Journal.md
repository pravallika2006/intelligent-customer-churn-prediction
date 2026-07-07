# Project Journal

## Day 1 – Phase 1: Business Understanding

### What I Learned
- Customer churn is...
- Businesses care about churn because...
- Machine Learning is used because...
- False Positives and False Negatives have different business impacts.

### Challenges
- Initially, I thought ML projects start with coding, but I learned that understanding the business problem comes first.

### Key Takeaways
- Every ML project begins with understanding the business objective.
- Accuracy alone does not define success.
- Machine Learning exists to solve business problems.

(i think i missed day-2 journal)

## Day 3 – Feature Analysis

### What I Learned

- EDA is not just about plotting graphs; it is about extracting meaningful business insights.
- Contract duration has a strong influence on customer churn.
- Fiber Optic customers unexpectedly have the highest churn rate.
- Data analysis should challenge assumptions rather than confirm them.
- Every business recommendation should balance customer satisfaction with company profitability.

### Business Thinking

While analyzing Fiber Optic customers, I realized that simply reducing prices may not always be the best solution. Improving customer support could reduce churn while maintaining revenue, making it a more sustainable strategy.

### Questions
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