import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.markdown("""
            Predict customer churn probability and identify at-risk customers.
            This tool hleps businesses reduce customer attrition through data-driven insights.
            """)
st.sidebar.header("Customer Details")

with st.sidebar.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Devce Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming Tv", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One-year", "two-year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic", "Mailed", "Bank Transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
    total_charges = st.slider("Total Charges ($)", 0, 9000, 2000)
    
    submitted = st.form_submit_button()
    
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Predictor")
    
    if submitted:
        with st.spinner("Predicting..."):
            try:
                customer_data = {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges
                }
                
                response = requests.post(
                    "https://farazgill-Telco-Customer-Churn-Fastapi.hf.space/predict",
                    json=customer_data
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.metric("Chrun Probability", f"{result['churn_probability']:.0%}")
                    st.metric("Prediction", "Churn" if result["churn_prediction"] == 1 else "Not Churn")
                    st.metric("Risk Level", result["risk_level"])
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["churn_probability"],  # make sure this is between 0–1
                        title={"text": "Churn Risk Gauge"},
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "darkblue", "thickness": 0.3},
                                "steps": [
                                    {"range": [0, 0.3], "color": "green"},
                                    {"range": [0.3, 0.7], "color": "yellow"},
                                    {"range": [0.7, 1], "color": "red"}
                                ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Error in Prediction. Please try again.")
                    
            except Exception as e:
                st.error(f"Could not contact to the API. Error: {e}")

with col2:
    st.header("Customer Analysis")
    
    if submitted:
        try:
            analysis_response = requests.get(
                "https://farazgill-Telco-Customer-Churn-Fastapi.hf.space/customer_analysis",
                params={
                    "tenure": tenure,
                    "monthly_charges": monthly_charges,
                    "contract": contract
                }
            )
            
            if analysis_response.status_code ==200:
                analysis = analysis_response.json()
                
                st.info("### Risk Factors")
                if analysis["risk_factors"]:
                    for factor in analysis["risk_factors"]:
                        st.write(f"⚠️ {factor}")
                else:
                    st.write("No significant risk factors indentified")
                    
                st.success("### Recommendations")
                for suggestion in analysis["suggestions"]:
                    st.write(f"✅ {suggestion}")
                    
            scatter_response = requests.get(
                "https://farazgill-Telco-Customer-Churn-Fastapi.hf.space/scatter_data"
            )
            
            if scatter_response.status_code == 200:
                scatter_df = pd.DataFrame(scatter_response.json())
                
                st.write("### Customer Position vs Typical Patterns")
                fig, ax = plt.subplots()
                
                churned = scatter_df[scatter_df["Churn"]==1]
                not_churned = scatter_df[scatter_df["Churn"]==0]
                
                ax.scatter(
                    churned["tenure"], churned["MonthlyCharges"],
                    alpha=0.6, color="red", label="Churned Customer"
                )
                
                ax.scatter(
                    not_churned["tenure"], not_churned["MonthlyCharges"],
                    alpha=0.6, color="green", label="Retained Customers"
                )
                
                ax.scatter(
                    tenure, monthly_charges,
                    color="blue", s=120, edgecolor="black", label="This Customer"
                )
                
                ax.set_xlabel("Tenure (months)")
                ax.set_ylabel("Monthly Charges ($)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("Could no fetch scatter plot data.")
                
        except Exception as e:
            st.error(f"Could not connect to the Api. Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
            *Data source: Kaggle Telco Customer Churn dataset. Predictions are estimates based on historical patterns.*

            """)
                
                
                
                    
                    
                
            
    
    
    
    
    

        











