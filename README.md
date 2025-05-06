# ATS Resume Matcher 🎯

An AI-powered tool that matches resumes to job descriptions, predicts candidate fit, and suggests tailored improvements.

![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Live Demo

🚀 [Launch the App](https://ats-resume-matcher.streamlit.app/)

---

## 📚 Project Overview

**ResumeMatcher** is an AI-driven application that:

- Predicts the job category based on a resume.
- Calculates the semantic similarity between a resume and a job description.
- Suggests missing technical keywords to optimize resumes for specific roles.

Built with **Python**, **Streamlit**, **spaCy**, and **Sentence-Transformers**.

---

## 🛠️ Features

- 📄 Upload your resume (PDF format).
- ✏️ Paste any job description.
- 📈 See real-time match score using semantic similarity.
- 🔍 Get smart suggestions to tailor your resume.
- 🧠 Predict your career category based on resume content.
- 🚀 Fully deployed and accessible via Streamlit Cloud.

---

## 🧰 Tech Stack

| Layer           | Technology                         |
|-----------------|-------------------------------------|
| Frontend UI     | Streamlit                          |
| Backend Models  | Python, Sentence-Transformers, scikit-learn |
| NLP Processing  | spaCy, PyPDF2                       |
| Cloud Hosting   | Streamlit Cloud                     |

---

## 🏗️ How It Works

1. Upload your resume (PDF).
2. Paste a job description into the app.
3. The app uses:
   - **Sentence-Transformers** to calculate semantic match.
   - **spaCy NER** to extract technical keywords dynamically.
   - **Random Forest Classifier** to predict the job category.
4. Displays:
   - Match Score (%)
   - Predicted Resume Category
   - Missing Skills Suggestions

---

## 🚀 Local Setup

To run the app locally:

```bash
# Clone this repo
git clone https://github.com/habid22/ResumeMatcher.git
cd ResumeMatcher

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
