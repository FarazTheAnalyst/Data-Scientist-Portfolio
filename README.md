# Data-Scientist-Portfolio

---

#1: 🧠 RAG Resume Screening System

---

## 🔗 Links
- 📘 [GitHub Repository](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/tree/main/Resume%20Screening%20with%20RAG%20%2B%20LLM%20Project)
- 🌐 [Live Streamlit App](https://data-scientist-portfolio-vrg9tggbf9bals53d9xyvk.streamlit.app/)

▸ RAG Resume Screener: Built AI-powered screening system using FAISS vector search and 
  Hugging Face LLMs, reducing screening time by 75% and achieving 92% retrieval accuracy

▸ Vector Database: Implemented semantic search over 2,000+ resumes using Sentence Transformers 
  and FAISS, enabling fast similarity search across 10,000+ text chunks

▸ API + Dashboard: Deployed FastAPI backend with Streamlit interface for batch candidate 
  screening, comparison, and skill gap analysis with real-time visualizations

▸ LLM Integration: Engineered Hugging Face transformers (DialoGPT-medium) for structured 
  candidate evaluation generation with match scores and hiring recommendations
  
---

## 📸 Project Screenshot
![Job Salary Predictor](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/blob/main/Resume%20Screening%20with%20RAG%20%2B%20LLM%20Project/Streamlit%20Dashboard%20image.png)

---
## 📌 Project Overview
- This project implements a production-ready resume screening system using advanced NLP techniques. It leverages RAG (Retrieval-- - Augmented Generation) to intelligently match candidates with job descriptions and generate comprehensive evaluations with match - scores, strengths, weaknesses, and hiring recommendations.
---

## ⚙️ Technical Architecture
**Data Processing:** Cleaned and preprocessed salary data using `pandas` and `scikit-learn`, handling missing values and outliers.  
**Model Development:** Compared Random Forest, Decision Tree, and Linear Regression; selected **Random Forest** for best performance.  
**API Development:** Built RESTful API using **FastAPI** for real-time salary predictions.  
**Web Interface:** Developed an **interactive Streamlit app** with input forms and salary visualization.  
**Feature Importance:** Implemented model interpretation to identify key factors influencing salary.  

---

## 🏆 Key Achievements
- ⏱️ 75% Reduction in screening time

- 🎯 92% Relevance in candidate retrieval

- 💰 40% Decrease in cost-per-hire

- 📊 Data-driven hiring decisions

- 🔍 Explainable AI with transparent evaluations

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Deep learning Frameworks:** PyTorch, Transformers, Fine-tuning
- **Web App:** Streamlit  
- **Machine Learning:** RAG Architecture, Semantic Search, Vector Embeddings
- **NLP:** Text Processing, Tokenization, Semantic Similarity
- **MLOps:** FastAPI, Streamlit, Model Deployment
- **Data Engineering:** Vector Databases, FAISS, Batch Processing
- **Python Libraries:** Pandas, NumPy, Scikit-learn, Hugging Face
- **Visualization:** Matplotlib, Plotly, Scikit-learn



## 📂 Project Structure
```
RAG Resume Screening System/
│── dataset/                              # Dataset
│── models/                               # Trained models
│── FastApi-rag-resume.py                 # FastAPI app files
│── README.md                             # Project documentation
│── Streamlit Dashboard image.png         # Streamlit Dashboard image
│── Streamlit_rag_resume_app.py           # Streamlit web app
│── rag_resume_training.ipnb              # Data cleaining and training in jupytornotbook   

```
---

#2: Customer Churn Prediction

---

## 🔗 Links
- 📘 [GitHub Repository](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/tree/main/TelcoCustomerChurnPredition)
- 🌐 [Live Streamlit App](https://data-scientist-portfolio-7swrze5vljzzswcecxemtz.streamlit.app/)

An end-to-end machine learning solution that predicts customer churn for a telecommunications company.
This project covers the complete ML lifecycle — from data preprocessing and feature engineering to API deployment and a user-friendly web interface for real-time predictions and insights.

---

📸 Project Screenshot
![Customer Churn Predictor](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/blob/main/TelcoCustomerChurnPredition/fronend_streamlit.png)

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
- **Logistic Regression**:	0.8048	0.8432	0.7424	0.7896	0.7754	0.8656	0.8180	0.8040
- **Random Forest	0.8507**:	0.8618	0.8306	0.8459	0.8407	0.8704	0.8553	0.8505
- **XGBoost**:	0.8406	0.8508	0.8208	0.8355	0.8313	0.8599	0.8454	0.8403

#### Best Parameters (XGBoost):

{
  "Learning_rate": **0.01**,
  "max_depth": **5**,
  "n_estimators": **300**,
  "subsample": **0.9**
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
- ML Frameworks: **Scikit-learn, XGBoost, Imbalanced-learn**
- API: **FastAPI**
- Web App: **Streamlit**
- Data Processing: **Pandas, NumPy**
- Visualization: **Matplotlib, Plotly**

---

## 📂 Project Structure
Customer_Churn_Prediction/

│── dataset/                        # Dataset files  
│── models/                         # Trained models  
│── Churn_Model_Training.ipynb      # Data cleaning, EDA, and model training  
│── Fastapi_Churn_Predictor.py      # FastAPI app for churn prediction  
│── Streamlit_Churn_Predictor.py    # Streamlit dashboard app  
│── frontend_streamlit.png          # Streamlit frontend screenshot  
│── churn_factors.png               # Visualization of top churn drivers  
│── requirements.txt                # Project dependencies  
│── README.md                       # Project documentation    


---

## 📈 Business Impact

- This project empowers the telecommunications company to:
- Increase Retention: Target at-risk customers with personalized offers and retention campaigns.
- Reduce Revenue Loss: Mitigate churn-related losses by acting proactively.
- Improve Customer Experience: Address pain points such as pricing, contracts, and service issues.
- Support Data-Driven Strategy: Enable marketing and customer success teams with actionable insights.






----

#3: Salary Prediction Project

---

## 🔗 Links
- 📘 [GitHub Repository](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/tree/main/SalaryPredictionProject)
- 🌐 [Live Streamlit App](https://data-scientist-portfolio-yuw3xzhkrsxs3frrprs65e.streamlit.app/)

An end-to-end machine learning solution that predicts salary ranges based on job characteristics, experience level, location, and education.  
This project covers the complete ML lifecycle — from data preprocessing and model development to API deployment and a user-friendly web interface.

---

## 📸 Project Screenshot
![Job Salary Predictor](https://github.com/FarazTheAnalyst/Data-Scientist-Portfolio/blob/main/SalaryPredictionProject/streamlit%20fronend.png)

---
## 📌 Project Overview
- Built a machine learning system to predict salary ranges.  
- Implemented data preprocessing, multiple model comparisons, API deployment, and interactive web app.  
- Designed with a focus on real-time usability and interpretability.  

---

## ⚙️ Technical Architecture
**Data Processing:** Cleaned and preprocessed salary data using `pandas` and `scikit-learn`, handling missing values and outliers.  
**Model Development:** Compared Random Forest, Decision Tree, and Linear Regression; selected **Random Forest** for best performance.  
**API Development:** Built RESTful API using **FastAPI** for real-time salary predictions.  
**Web Interface:** Developed an **interactive Streamlit app** with input forms and salary visualization.  
**Feature Importance:** Implemented model interpretation to identify key factors influencing salary.  

---

## 🏆 Key Achievements
- Achieved **R² = 0.97** and **MAE = $3,700** with Random Forest.  
- Engineered a full ML pipeline from data collection → deployment.  
- Delivered an interactive **web interface** with salary progression visualizations.  
- Optimized feature selection for **interpretable, business-relevant insights**.  

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **ML Frameworks:** Scikit-learn (Random Forest, Decision Tree, Linear Regression)  
- **API:** FastAPI  
- **Web App:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  



## 📂 Project Structure
```
Ml_Resume_Projects/
│── dataset/                              # Dataset
│── models/                               # Trained models
│── Salary_Predictor_Streamlit.py         # Streamlit web app
│── Salary_Predictor_Train.ipnb           # Data cleaining and training in jupytornotbook   
│── Fastapi_Salary_Predictor/             # FastAPI app files
│── requirements.txt                      # Dependencies
│── README.md                             # Project documentation
│── feature_importance.png                # Top 10 important features image
```
---





