# Dataset Understanding

## Dataset Name
IBM Telco Customer Churn Dataset

## Source
Kaggle (BlastChar)

## Domain
Telecommunications

## Problem Type
Binary Classification

## Target Variable
Churn

Values:
- Yes
- No

## Number of Records
7043

## Number of Features
21 Columns (20 Features + 1 Target)

## Business Context

The dataset contains information about telecom customers including demographic details, subscribed services, billing information, tenure, and whether they left the company.

The objective is to predict customers who are likely to churn so that the business can take proactive retention actions.

## Initial Observations

- The dataset contains 7043 customer records and 21 columns.
- The target variable is **Churn**, making this a binary classification problem.
- Most features are categorical.
- `SeniorCitizen`(0 or 1) is stored as an integer but represents a categorical variable.
- `TotalCharges`(ex: 1889.5) is stored as an object despite appearing to contain numeric values, which requires further investigation.
- No null values are reported by `df.info()`, but additional validation is needed to detect blank strings or other hidden missing values.

## Data Quality Assessment

### Missing Values
- No null values were detected using `isnull()`.
- The `TotalCharges` column contains hidden missing values represented as blank strings.

### Duplicate Records
- No duplicate rows were found.
- No duplicate customer IDs were found.

### Unique Values
- `customerID` has 7043 unique values, confirming each customer is unique.
- Most categorical features contain between 2 and 4 categories.
- `MonthlyCharges` has 1585 unique values.
- `TotalCharges` has 6531 unique values because customers accumulate different total amounts over time.

### Business Observation
Customers with blank `TotalCharges` have `tenure = 0`, indicating they are new customers who have not completed their first billing cycle. This explains why their total billed amount is not yet available.

### Data Cleaning

- Converted TotalCharges from object to float.
- Found blank values in TotalCharges.
- Blank values belonged to customers with tenure = 0.
- Replaced blank values with 0.
- No missing values remain.

---

## Class Distribution

- No : 5174 (73.46%)
- Yes : 1869 (26.54%)

Dataset is moderately imbalanced.

---

## Key EDA Findings

- Longer tenure → Lower churn.
- Month-to-month contracts have the highest churn.
- Higher MonthlyCharges correlate with higher churn.
- Fiber optic customers churn more than DSL customers.
- Electronic check users have the highest churn.
- Customers without OnlineSecurity or TechSupport churn much more.
- Customers with Partner or Dependents churn less.
- SeniorCitizen customers churn more.
- StreamingTV and StreamingMovies have minimal impact.
- Gender has almost no effect.

---

## Features to Drop

- customerID (Unique identifier with no predictive value.)