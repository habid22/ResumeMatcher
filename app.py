# 🚀 Fully Updated Streamlit App - Resume Matcher with Dynamic Suggestions (NER Model)

import streamlit as st
from utils.resume_parser import extract_text_from_pdf
from utils.model_utils import calculate_match_score, predict_category, suggest_resume_improvements_dynamic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Resume Matcher", layout="wide")

# --- Header ---
st.title("\U0001F4BC AI Resume Matcher & Career Advisor")
st.markdown("Effortlessly match your resume with any job description and get career insights!")

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Resume Matcher", "About Project"])

if page == "Resume Matcher":

    # --- Upload Resume ---
    st.subheader("Upload Your Resume")
    resume_file = st.file_uploader("Choose your resume (PDF format)", type=["pdf"])

    resume_text = ""
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        st.success("Resume uploaded and parsed successfully!")

    # --- Paste Job Description ---
    st.subheader("Paste Job Description")
    job_description = st.text_area("Paste the job description you want to match with")

    # --- Run Analysis ---
    if st.button("Analyze Match"):
        if resume_text and job_description:
            with st.spinner("Analyzing..."):
                match_score = calculate_match_score(resume_text, job_description, model)
                predicted_role = predict_category(resume_text)
                missing_keywords = suggest_resume_improvements_dynamic(job_description, resume_text)

            # --- Results ---
            st.subheader("\U0001F4CA Results")

            # Show Match Score
            st.progress(match_score / 100)
            st.metric(label="Match Score", value=f"{match_score}%")

            # Interpretation
            if match_score >= 80:
                st.success("Excellent match! 🚀")
            elif match_score >= 60:
                st.info("Good match. Some improvements needed.")
            else:
                st.warning("Weak match. Tailor your resume.")

            # Predicted Category
            st.subheader("\U0001F4D6 Predicted Resume Category")
            st.markdown(f"### 🎯 `{predicted_role}`")

            # Suggestions for Improvement
            st.subheader("🛠 Suggestions to Tailor Your Resume")
            if missing_keywords:
                st.write("Consider adding or emphasizing these dynamically identified skills, tools, or technologies to better align your resume:")
                st.markdown(", ".join(f"`{word}`" for word in missing_keywords))
            else:
                st.success("Your resume already covers most critical skills! 🚀")

        else:
            st.error("Please upload a resume and paste a job description.")

elif page == "About Project":
    st.subheader("About Resume Matcher")
    st.write("""
    **Resume Matcher** is an AI-powered Streamlit app designed to:
    - Upload your resume (PDF)
    - Match it against job descriptions
    - Predict your career category
    - Offer resume optimization advice

    Built using:
    - Streamlit
    - Sentence Transformers
    - Random Forest Classifier
    - TF-IDF Vectorization

    Ideal for career advisors, job seekers, and recruiters.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with 💬 by Hassan Amin| Demo Project 2025")
