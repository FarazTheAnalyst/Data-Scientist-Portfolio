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
model = joblib.load("churn_predictor_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")
numerical_cols = joblib.load("numerical_cols.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
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
    try:
        # existing code...
        input_dict = customer_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Feature engineering
        input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=[-1, 12, 24, 48, 72, np.inf],
                                         labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"])

        input_df["MonthlyChargeGroup"] = pd.cut(input_df["MonthlyCharges"],
                                                bins=[0, 35, 70, 90, np.inf],
                                                labels=["Low", "Medium", "High", "VeryHigh"],
                                                include_lowest=True)
        
        input_df["chargeRatio"] = input_df["TotalCharges"] / input_df["MonthlyCharges"]
        input_df["chargeRatio"] = input_df["chargeRatio"].replace([np.inf, -np.inf], 0)

        # Encoding
        for col in categorical_cols:
            if col in input_df.columns:
                le = label_encoders[col]
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df[col] = le.transform(input_df[col])

        # Scaling
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Align with training features
        X = input_df[feature_cols]

        churn_prob = model.predict_proba(X)[0][1]
        churn_pred = int(model.predict(X)[0])

        return {
            "churn_probability": round(float(churn_prob), 3),
            "churn_prediction": churn_pred,
            "risk_level": "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.4 else "Low"
        }

    except Exception as e:
        return {"error": str(e)}

    

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
            
