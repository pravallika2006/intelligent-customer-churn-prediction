# SHAP Analysis

## Objective

Machine Learning models such as Random Forest provide accurate predictions but are often difficult to interpret. To improve transparency and explainability, SHAP (SHapley Additive exPlanations) was integrated into this project.

---

## Why SHAP?

SHAP explains how each feature contributes to an individual prediction.

Instead of only predicting whether a customer is likely to churn, SHAP answers:

- Why was this prediction made?
- Which features increased churn risk?
- Which features reduced churn risk?

This makes the model more interpretable for business users.

---

## Implementation

A SHAP TreeExplainer was created using the trained Random Forest model.

```python
explainer = shap.TreeExplainer(model)
```

For every customer prediction:

1. Customer data is preprocessed.
2. SHAP values are calculated.
3. A visualization is generated to explain the prediction.

---

## Visualization

The Streamlit application displays a SHAP feature importance chart after every prediction.

The chart shows:

- Features pushing the prediction toward churn
- Features reducing churn probability
- Relative contribution of each feature

This enables users to understand the reasoning behind the model instead of treating it as a black box.

---

## Benefits

- Improves model transparency
- Builds trust in predictions
- Helps customer retention teams understand churn drivers
- Supports explainable AI principles

---

## Conclusion

By integrating SHAP, the Customer Churn Prediction System provides not only accurate predictions but also meaningful explanations, making the solution more useful for business decision-making.