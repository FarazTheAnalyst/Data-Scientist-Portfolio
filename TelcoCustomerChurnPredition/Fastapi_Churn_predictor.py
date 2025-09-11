from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability"
)

# Load the trained model and preprocessing objects
model = joblib.load("E:\Project Ml\Telco Customer Churn\model\churn_predictor_model.pkl")
label_encoders = joblib.load("E:\Project Ml\Telco Customer Churn\model\label_encoders.pkl")
scaler = joblib.load("E:\Project Ml\Telco Customer Churn\model")
feature_cols = joblib.load("E:\Project Ml\Telco Customer Churn\model\feature_cols.pkl")
categorical_cols = joblib.load("E:\Project Ml\Telco Customer Churn\model\categorical_cols.pk")
numerical_cols = joblib.load("E:\Project Ml\Telco Customer Churn\model\numerical_cols.pkl")

class CustomerData(BasemModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int 
    PhoneService: str
    MultipleLines: str
    internetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamlingTv: str
    StreamlingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    
@app.get("/")
def read_root():
    return{"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Convert input data to dataframe
    input_dict = customer_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Apply feature engineering (same as training)
    input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=[0, 12, 24, 48, 72, np.inf],
                    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"])
                    
    input_df["MonthlyChargeGroup"] = pd.cut(input_df["MonthlyCharges"], bins[0, 35, 70, 90, np.inf],
                                        labels=["Low", "Medium", "High", "VeryHigh"])
                                        
    # Encode categorical variables
    for col in categorical_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            # Handle unseen labels
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])
            
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[col])
    
    # Ensure all columns are in the right order
    X = input_df[feature_cols]
    
    # Make predictoins
    churn_prob = model.predict_proba(X)[0][1]
    churn_pred = int(model.predict(X)[0])
    
    return {
        "churn_probability": round(float(churn_prob), 3),
        "churn_prediction": churn_pred,
        "risk_levl": "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.4 else "Low"
    }
    

@app.get("/customer_analysis")
def customer_analysis(tenure: int, monthly_charges: float, contract: str):
    # This end point provides insights without predictions
    risk_factors = []
    
    if tenure < 12:
        risk_factors.append("Low tenure (new customer)")
    if monthly_charges > 70:
        risk_factors.append("High Monthly Charges")
    if  contract == "Month-to-month":
        risk_factors.append("Month-to-month contract")
        
    return {
        "risk_factors": risk_factors,
        "suggestions": [
            "Consider offering a loyalty discount",
            "Recommand annual contract for better rates",
            "Check if customer needs technical support"
        ] if risk_factors else ["Customer profile indicates low churn risk"]
    }           
            