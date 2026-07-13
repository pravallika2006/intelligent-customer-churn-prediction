import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/best_model.pkl")

explainer = shap.Explainer(model)


def explain_prediction(customer_df):

    shap_values = explainer(customer_df)

    values = shap_values.values[0]

    # If model returns values for two classes,
    # use the Churn class (index 1)
    if values.ndim == 2:
        values = values[:, 1]

    feature_names = customer_df.columns

    # Top 10 important features
    idx = np.argsort(np.abs(values))[-10:]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        feature_names[idx],
        values[idx]
    )

    ax.set_title("Top Factors Influencing This Prediction")

    plt.tight_layout()

    return fig