from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Job Salary Predictor API",
    description="API for predicting salaries based on job characteristics"
)

# Load the model
model = joblib.load(r"E:\Salary_Prediction_Project\salary_predictor.pkl")

class PredictionInput(BaseModel):
    Job_Title: str
    Years_of_Experience: float
    Education_Level: str
    Gender: str

#root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Job Salary Predictor API"}

#predict
@app.post("/predict")
def predict_salary(input_data: PredictionInput):
  #Prepare input data
  input_df = pd.DataFrame([{
      "Job_Title": input_data.Job_Title,
      "Years_of_Experience": input_data.Years_of_Experience,
      "Education_Level": input_data.Education_Level,
      "Gender": input_data.Gender
  }])

  #Make Predictions
  predictions = model.predict(input_df)

  return {
      "predicted_salary" : round(float(predictions[0]), 2),
      "currency" : "USD"
  }
  
if __name__ == "__main__":
    import uvicorn
    uvicorm.run(app, host="0.0.0.0", port=7860)

