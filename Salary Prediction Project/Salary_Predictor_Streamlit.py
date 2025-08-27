import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set page config
st.set_page_config(
    page_title="Job Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)



#Title and description
st.title("💰 Job Salary Predictor")
st.markdown("Predict the expected salaries based on experience, job title, gender, education level.")

st.sidebar.subheader("Top Importent Features")
st.sidebar.image("Ml_Resume_Projects/feature_importance.png", caption="Top features")


#Main form
col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input("Job Title", "Data Scientist")
    experience = st.number_input("Year of Experience", min_value=0, max_value=50, value=5)
    
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "High School"])

#url
#ngrok_url = "https://d7fcfd3e9ec8.ngrok-free.app/predict"

#Predict button
if st.button("Predict Salary"):
    with st.spinner("Predicting..."):
        #call FastAPI endpoint
        try:
            response = requests.post(
                "https://farazgill-salary-predictor-fastapi.hf.space/predict",
                json={
                    "Job_Title": job_title,
                    "Years_of_Experience": experience,
                    "Education_Level": education,
                    "Gender": gender
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Salary: ${result['predicted_salary']:,.2f} {result['currency']}")

                
                #show some insights
                st.subheader("Salary Insights")
                
                #Create a hypothetical salary progression
                years = list(range(0, 21, 5))
                salaries = []
                
                for year in years:
                    resp = requests.post(
                        "https://farazgill-salary-predictor-fastapi.hf.space/predict",
                        json={
                            "Job_Title": job_title,
                            "Years_of_Experience": year,
                            "Education_Level": education,
                            "Gender": gender
                        }
                    )
                    
                    if resp.status_code == 200:
                        salaries.append(resp.json()["predicted_salary"])
                        
                #Plot salary progression
                sns.set_theme(style="whitegrid")
                sns.set_palette("pastel")
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(years, salaries, marker="o", linewidth=3, color="#2E86AB", markersize=6)
                ax.set_xlabel("Years of Experience", fontsize=12)
                ax.set_ylabel("Predicted Salary (USD)", fontsize=12)
                ax.set_title("Salary Growth Projection")
                ax.grid(True, linestyle="--", alpha=0.9)
                sns.despine()
                st.pyplot(fig)
    
            else:
                st.error("Error in prediction. Please try again.")
        except Exception as e:
            st.error(f"Could not connect to the API. Error: {e}")
    
#Sample data section
st.subheader("Sample Salary Data")
sample_data = pd.DataFrame({
    "job_title": ["Data Scientist", "Software Engineer"],
    "Experience": [5, 3],
    "Gender": ["Male", "Female"],
    "Education": ["Master's", "Bachelor's"],
    "Salary": [120000, 110000]
})

st.dataframe(sample_data)

#Footer
st.markdown("---")
st.markdown("""
            Data sourced from kaggle: Predictions are estimates based on historical data.
            """)

