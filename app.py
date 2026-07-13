import streamlit as st
import pandas as pd
import joblib

import plotly.graph_objects as go
import plotly.express as px

from src.explain import explain_prediction
# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


with st.sidebar:

    st.image(
        "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/streamlit.svg",
        width=80
    )

    st.title("Telecom Churn")

    st.markdown("---")

    st.markdown("### Project")

    st.write("Customer Churn Prediction")

    st.markdown("### Model")

    st.success("Random Forest (Tuned)")

    st.markdown("### 📊 Model Performance")

    st.metric("Accuracy", "75.73%")
    st.metric("Recall", "79.68%")
    st.metric("F1 Score", "63.54%")
    st.metric("ROC-AUC", "76.99%")

    st.markdown("### Developer")

    st.write("Devagurthi Teja  Pravallika")

    st.markdown("---")

    st.caption("Machine Learning Internship Project")
# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# =====================================================
# HEADER
# =====================================================

st.title("📊 Intelligent Customer Churn Prediction")

st.markdown(
    """
Predict whether a telecom customer is likely to churn using a trained Machine Learning model.

"""
)

st.divider()

# =====================================================
# MAIN LAYOUT
# =====================================================

left_column, right_column = st.columns([1, 1.2])

# =====================================================
# LEFT PANEL
# =====================================================

with left_column:

    st.subheader("📝 Customer Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    st.divider()

    st.subheader("📞 Services")

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    st.divider()

    st.subheader("💳 Billing")

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )

    st.divider()

    predict_button = st.button(
        "🚀 Predict Churn",
        use_container_width=True,
        type="primary"
    )

    # =====================================================
# RIGHT PANEL
# =====================================================

with right_column:

    st.subheader("📈 Prediction Dashboard")

    if not predict_button:

        st.info(
            """
### 👋 Welcome!

Please complete the customer information form on the left and then click **🚀 Predict Churn**.

Once you submit the details, this dashboard will display:

- 📊 Churn Probability
- 🎯 Prediction Result
- ⚠ Risk Level
- 💼 Business Insights
- 📈 Feature Importance
- 💡 Recommended Retention Actions
"""
        )

    else:

        # ==========================================
        # CREATE CUSTOMER DATAFRAME
        # ==========================================

        customer = pd.DataFrame([{

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges

        }])

        # ==========================================
        # ONE HOT ENCODING
        # ==========================================

        customer = pd.get_dummies(customer)

        customer = customer.reindex(
            columns=feature_columns,
            fill_value=0
        )

        numeric_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        customer[numeric_cols] = scaler.transform(
            customer[numeric_cols]
        )

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = model.predict(customer)[0]
        probability = model.predict_proba(customer)[0][1]

        risk = probability * 100

        # ==========================================
        # KPI CARDS
        # ==========================================

        c1, c2, c3 = st.columns(3)

        with c1:

            if prediction == 1:

                st.metric(
                    "Prediction",
                    "⚠ Churn"
                )

            else:

                st.metric(
                    "Prediction",
                    "✅ Stay"
                )

        with c2:

            if risk < 30:

                risk_level = "Low"

                st.metric(
                    "Risk Level",
                    "Low"
                )

            elif risk < 60:

                risk_level = "Medium"

                st.metric(
                    "Risk Level",
                    "Medium"
                )

            else:

                risk_level = "High"

                st.metric(
                    "Risk Level",
                    "High"
                )
        with c3:

            st.metric(
                "Probability",
                f"{risk:.2f}%"
            )

        st.divider()

        # ==========================================
        # GAUGE CHART
        # ==========================================

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                number={
                    "suffix": "%"
                },
                title={
                    "text": "Churn Probability"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "crimson"
                    },

                    "steps": [

                        {
                            "range": [0, 30],
                            "color": "#90EE90"
                        },

                        {
                            "range": [30, 60],
                            "color": "#FFD966"
                        },

                        {
                            "range": [60, 100],
                            "color": "#FF9999"
                        }

                    ]

                }
            )
        )

        gauge.update_layout(
            height=320,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ==========================================
        # CUSTOMER PROFILE
        # ==========================================

        st.subheader("👤 Customer Profile")

        profile_left, profile_right = st.columns(2)

        with profile_left:

            st.markdown(f"**Gender:** {gender}")
            st.markdown(f"**Tenure:** {tenure} Months")
            st.markdown(f"**Contract:** {contract}")
            st.markdown(f"**Internet:** {internet}")

        with profile_right:

            st.markdown(f"**Monthly Charges:** ${monthly_charges:.2f}")
            st.markdown(f"**Tech Support:** {tech_support}")
            st.markdown(f"**Paperless Billing:** {paperless}")
            st.markdown(f"**Payment:** {payment_method}")

        st.divider()

        # ==========================================
        # BUSINESS INSIGHT
        # ==========================================

        st.subheader("💼 Business Insight")

        insights = []

        if contract == "Month-to-month":
            insights.append(
    "Month-to-month contracts historically have the highest churn."
)

        if tenure < 12:
            insights.append(
                "Customer is relatively new, making churn more likely."
            )

        if monthly_charges > 70:
            insights.append(
                "High monthly charges may increase churn risk."
            )

        if tech_support == "No":
            insights.append(
                "Lack of Tech Support is associated with higher churn."
            )

        if internet == "Fiber optic":
            insights.append(
                "Customers using Fiber Optic internet have historically shown higher churn rates."
            )

        if len(insights) == 0:

            st.success(
                "This customer shows relatively stable characteristics based on historical data."
            )

        else:

            for item in insights:
                st.info(item)
        

        st.divider()

        # ==========================================
        # RETENTION ACTIONS
        # ==========================================

        st.subheader("🎯 Recommended Retention Actions")

        recommendations = []

        if contract == "Month-to-month":
            recommendations.append(
                "Offer One-Year Contract Discount"
            )

        if tech_support == "No":
            recommendations.append(
                "Offer a complimentary Tech Support trial."
            )

        if monthly_charges > 70:
            recommendations.append(
                "Offer Personalized Pricing Plan"
            )

        if tenure < 12:
            recommendations.append(
                "Enroll Customer in Loyalty Program"
            )

        if probability >= 0.60:
            recommendations.append(
                "Schedule Customer Success Follow-up"
            )

        if len(recommendations) == 0:

            st.success(
                "No immediate retention action is required."
            )

        else:

            for item in recommendations:

                st.success(f"✔ {item}")

        st.divider()


        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        st.subheader("📊 Top Features Influencing Prediction")

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({

                "Feature": feature_columns,
                "Importance": model.feature_importances_

            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            fig = px.bar(

                importance.sort_values("Importance"),

                x="Importance",
                y="Feature",

                orientation="h",

                text="Importance",

                color="Importance",

                color_continuous_scale="Blues"

            )

            fig.update_layout(

                height=420,

                coloraxis_showscale=False,

                xaxis_title="Importance Score",

                yaxis_title="",

                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=20
                )

            )

            fig.update_traces(

                texttemplate="%{text:.3f}",

                textposition="outside"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==========================================
            # SHAP EXPLAINABILITY
            # ==========================================

            st.divider()

            st.subheader("🔍 Why did the model make this prediction?")

            with st.spinner("Generating explanation..."):

                shap_fig = explain_prediction(customer)

                st.pyplot(shap_fig)
                st.info(
                    """
                This chart explains which customer attributes had the strongest influence on the prediction.

                • Positive values increase churn risk.
                • Negative values reduce churn risk.
                """
                )

        else:

            st.info(
                "Feature importance is not available for this model."
            )


        
st.divider()