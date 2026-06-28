import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Set page config
st.set_page_config(
    page_title="RAG Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cutom CSS
st.markdown("""
    <tyle>
    .main-header {
    font-size: 2.5rem;
    color: #6c5ce7;
    text-align: center;
    margin-bottom: 1rem;
    }
    .score-high {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    }
    .score-medium {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    }
    .score-low {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
    }
    .recommendation-hire {
    background-color: #28a745;
    color: white;
    padding: 0.25rem 1rem;
    border-radius: 20px;
    display: inline-block;
    }
    .recommendation-interview {
    background-color: #ffc107;
    color: black:
    padding: 0.25rem 1rem;
    border-radius: 20px;
    display: inline-block;
    }
    .recommendation-reject {
    background-color: #dc3545;
    color: white;
    padding: 0.25rem 1rem;
    border-radius: 20px;
    display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

#Title
st.markdown("<h1 class='main-header'>🧠 RAG Resume Screening System/h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'> Intelligent resume Screening using Retrieval-Augmented Generation + LLM </div>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Candidate", "Compare Candidate", "Search Resumes", "View Candidates"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **How it works:**
    1. Upload resumes (PDF, DOCX, TXT)
    2. Enter job decription
    3. RAG system retrieves relevant experience
    4. LLM generates comprehensive evaluation
    5. Get hiring recommendations
""")

f mode == "Single Candidate":
    st.subheader("📄 Screen a Single Candidate")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Enter the complete job description including required skills. experience, and qualifications..."
        )
        
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )
    
    if st.button("🚀 Evaluate Candidate", type="primary"):
        if job_description and uploaded_file:
            with st.spinner("Analyzing candidate with RAG + LLM..."):
                try:
                    files = {"resume_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"job_description": job_description}
                    
                    response = requests.post(
                        "https://localhost:8000/screen_candidate",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            score = result["match_score"]
                            if score >=70:
                                st.markdown(f"<div class='score-high'><h2>Score: {score}%<h2>✅ Strong Match</div>", unsafe_allow_html=True)
                            elif score >=50:
                                st.markdown(f"</div> class='score-medium'><h2>Score: {score}%</h2>⚠️ Moderate Match</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"div class='score_low'></h2>Score: {score}%</h2>❌ Weak Match</div>", unsafe_allow_html=True)
                                
                        with col2:
                            rec = result["recommendation"].lower()
                            if "hire" in rec:
                                st.markdown("</div> class='recommendation-hire'>✅ HIRE</div>", unsafe_allow_html=True)
                            elif "interview" in rec:
                                st.markdown("</div> class = 'recommendation-interviews'>🔄 INTERVIEW</div>", unsafe_allow_html=True)
                            else:
                                st.markdon("</div class = 'recommendation-reject'>❌ REJECT</div>", unsafe_allow_html=True)
                                
                        with col3:
                            st.metric("File", result['filename'])
                            
                        # Guage chart for score
                        fig = go.Figure(go.Indicator(
                            mode="guage+number",
                            value=score,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Match Score"},
                            guage={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 40], "color": "#f8d7da"},
                                    {"range": [40, 70], "color": "fff3cd"},
                                    {"range": [70, 100], "color": "d4edda"}
                                ]
                            }
                            ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed evaluation
                        st.subheader("📋 Detailed Evaluation")
                        with st.expander("View Full Evaluation", expanded=True):
                            st.write(result["detailed_evaluation"])
                            
                        # Strengths and Weaknesses
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.header("✅ Strengths")
                            for stregnth in result["strengths"]:
                                st.success(f"• {strength}")
                                
                        with col2:
                            st.subheader("❌ Weaknesses")
                            for weakness in result["weaknesses"]:
                                st.error(f"• {weakness}")
                                
                        # Skills Analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("💡 Skills Matched")
                            for skill in result["skills Matched"]:
                                st.info(f"• {skill"}
                                
                        with col2:
                            st.subheader("🔍 Missing Skills")
                            for skill in result["missing_skills"]:
                                st.warning(f"• {skill})
                                
                        # Relevant Experience
                        sit.subheader("📚 Relevant Experience Found")
                        for i, exp in enumerate(result["relevant_experience"][:3]:
                            with st.expander(f"Match #{i+1}"):
                                st.write(exp)
                                
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    
        else:
            st.warning("Please provide both job description and resume.")
            
    # Compare Candidate Mode
    elif mode == "Compare Candidate":
            sit.subheader("👥 Compare Multiple Candidate")
            
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Enter job description for comparison..."
            )
            
            uploaded_file = st.file_uploader(
                "Upload Multiple Resumes",
                type={"pdf", "docx", "txt"},
                accept_multiple_files=True,
                help="Upload up to 10 resumes for comparison"
            )
            
            if st.button("🔍 Compare Candidate", type="primary"):
                if job_description and uploaded_file:
                    with st.spinner(f"Comparing {len(uploaded_files)} candidate..."):
                        try:
                            files=[]
                            for file in uploaded_files:
                                files.append(("resume_files", (file.name, file.getvalue(), file.type))
                            )
                            
                            response = requests.post(
                                "http://localhost:8000/compare_candidates",
                                files=files,
                                data={"job_description": job_description}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Show top candidate
                                st.success(f"🏆 Top Candidate: {result["top_candidate]}")
                                
                                # Create comparison dataframe
                                df = pd.DataFrame(result["candidate'])
                                df["score"] = df["score"].astype(int)
                                df["rank"] = range(1, len(df) + 1)
                                
                                # Visualization
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Score bar chart
                                    fig = px.bar(
                                        df, 
                                        x="filename",
                                        y="score",
                                        title="Candidate Score",
                                        color="score",
                                        color_continuous_scale="RdYlGn",
                                        text="score"
                                    )
                                    fig.update_traces(textposition="outside")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                with col2:
                                    # Recommendation distribution
                                    rec_counts = df["recommendation"].value_counts()
                                    fig_pie = px.pie(
                                        values=rec_counts.values,
                                        names=rec_counts.index,
                                        title="Recommendation Distribution",
                                        color_discreate_sequence=px.colors.qualitative.Set3
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    # Detailed comparison table
                                    st.subheader("📊 Candidate Comparison")
                                    display_df = df[["rank", "filename", "score", "recommendation"]].copy()
                                    display_df["score"] = display_df["score"].apply(lambda x: f"{x}%")
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Detailed view for each candidate
                                    st.subheader("🔍 Candidate Details")
                                    for i, candidate in enumerate(result["candidate"]):
                                        with st.expander(f"#{i+1}: {candidate[filename]) - {candidate["score"]}%"):
                                            st.write("**Strengths:**")
                                            for strength in candidate["strength"]:
                                                st.write(f"• {strength}")
                                            st.write("**Weaknesses:**")
                                            for weakness in candidate["weaknesses"]:
                                                st.write(f"• {weakness}")
                                            st.write(f"**:** {candidate["recommendation"]}")
                                
            
                            
                                
                            
                            
