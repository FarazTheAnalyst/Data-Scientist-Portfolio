Customer Churn Prediction
---

## 🔗 Links
- 📘 GitHub Repository
- 🌐 Live Streamlit App

An end-to-end machine learning solution that predicts customer churn for a telecommunications company.
This project covers the complete ML lifecycle — from data preprocessing and feature engineering to API deployment and a user-friendly web interface for real-time predictions and insights.

---

📸 Project Screenshot
![Customer Churn Predictor](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/blob/main/TelcoCustomerChurnPredition/streamlit%20fronend.png)

--

## 📌 Project Overview

- Built a machine learning system to predict customer churn.
- Implemented data preprocessing, feature engineering, and handled class imbalance.
- Compared multiple algorithms (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning.
- Deployed an API with FastAPI and an interactive Streamlit dashboard with business insights.

---

## ⚙️ Technical Architecture

- Data Processing
- Cleaned missing values and encoded categorical variables
- Addressed class imbalance (73:27 ratio) using SMOTE
- Feature Engineering
- Created new features (e.g., tenure groups, payment categories) to boost model performance
- Model Development
- Trained and compared:
- Logistic Regression
- Random Forest
- XGBoost
- Performed hyperparameter tuning for XGBoost

## 🍨 Model Comparison Results

- **Model	Accuracy	Precision (0)	Recall (0)	F1 (0)	Precision (1)	Recall (1)	F1 (1)	AUC**
- **Logistic Regression	0.8048	0.8432	0.7424	0.7896	0.7754	0.8656	0.8180	0.8040**
- **Random Forest	0.8507	0.8618	0.8306	0.8459	0.8407	0.8704	0.8553	0.8505**
- **XGBoost	0.8406	0.8508	0.8208	0.8355	0.8313	0.8599	0.8454	0.8403**

#### Best Parameters (XGBoost):

{
  "Learning_rate": 0.01,
  "max_depth": 5,
  "n_estimators": 300,
  "subsample": 0.9
}

**Best AUC Score: 0.9067**

- API Development
- Developed a RESTful API with FastAPI for real-time churn predictions
- Web Interface
- Built an interactive Streamlit dashboard with churn probability visualization and business insights

---

## 🏆 Key Achievements

- Handled severe class imbalance with SMOTE
- Achieved 82% precision and 78% recall on churned customers (minority class)
- Identified key churn factors:
- Contract Type
- Tenure
- Monthly Charges
- Delivered an intuitive dashboard for scenario testing & interventions


---

## 🛠️ Tech Stack

Programming: Python
ML Frameworks: Scikit-learn, XGBoost, Imbalanced-learn
API: FastAPI
Web App: Streamlit
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Plotly

---

## 📂 Project Structure
Customer_Churn_Prediction/
│── dataset/                           # Dataset files  
│── models/                            # Trained models  
│── Churn_Model_Training.ipynb         # Data cleaning, EDA, model training  
│── Fastapi_Churn_predictor.py/         # FastAPI app files  
│── README.md                          # Project documentation   
│── Streamlit_churn_predictor.py       # Streamlit dashboard files 
│── requirements.txt                   # Dependencies  
│── churn_factors.png                  # Top churn drivers visualization  

---

## 📈 Business Impact

- This project empowers the telecommunications company to:
- Increase Retention: Target at-risk customers with personalized offers and retention campaigns.
- Reduce Revenue Loss: Mitigate churn-related losses by acting proactively.
- Improve Customer Experience: Address pain points such as pricing, contracts, and service issues.
- Support Data-Driven Strategy: Enable marketing and customer success teams with actionable insights.
