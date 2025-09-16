Customer Churn Prediction - Machine Learning Project
📌 Project Overview

This project develops a machine learning solution to predict customer churn for a telecommunications company. The solution focuses on handling class imbalance and generating actionable insights to help reduce customer attrition.

Dataset: Telco Customer Churn (Kaggle)

Business Goal: Identify customers at high risk of leaving and suggest intervention strategies.

⚙️ Technical Architecture
🔹 Data Processing

Handled missing values and outliers

Encoded categorical variables

Addressed class imbalance (73:27) using SMOTE

🔹 Feature Engineering

Created meaningful features such as tenure groups, payment behavior, and contract types

🔹 Model Development

Compared multiple ML algorithms:

Logistic Regression

Random Forest

XGBoost

Performed hyperparameter tuning for XGBoost

🔹 Evaluation Metrics

Precision, Recall, F1-score, AUC-ROC used (class imbalance aware)

🔹 Deployment

API Development: Built a RESTful API with FastAPI for real-time churn predictions

Web Interface: Created an interactive Streamlit dashboard with churn visualizations & business insights

📊 Model Comparison Results
Model	Accuracy	Precision (0)	Recall (0)	F1 (0)	Precision (1)	Recall (1)	F1 (1)	AUC
Logistic Regression	0.8048	0.8432	0.7424	0.7896	0.7754	0.8656	0.8180	0.8040
Random Forest	0.8507	0.8618	0.8306	0.8459	0.8407	0.8704	0.8553	0.8505
XGBoost	0.8406	0.8508	0.8208	0.8355	0.8313	0.8599	0.8454	0.8403

Best Hyperparameters for XGBoost:

{
  "Learning_rate": 0.01,
  "max_depth": 5,
  "n_estimators": 300,
  "subsample": 0.9
}


Best Score (AUC): 0.9067

🏆 Key Achievements

Successfully handled severe class imbalance using SMOTE

Achieved 82% precision and 78% recall on minority (churn) class using XGBoost

Identified top churn drivers:

Contract type

Tenure

Monthly charges

Built a Streamlit dashboard enabling business users to simulate intervention strategies

🛠️ Technical Stack

Programming: Python

ML Frameworks: Scikit-learn, XGBoost, Imbalanced-learn

API Framework: FastAPI

Web Framework: Streamlit

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Plotly

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Install dependencies:

pip install -r requirements.txt


Train models:

python train.py


Run API server:

uvicorn api:app --reload


Launch Streamlit dashboard:

streamlit run dashboard.py

📈 Business Impact

By identifying at-risk customers and their churn drivers, this project enables the telecom company to:

Improve customer retention strategies

Reduce churn-related revenue loss

Enhance targeted marketing campaigns
