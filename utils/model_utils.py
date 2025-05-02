import numpy as np
import pickle

# Load trained model and vectorizer
try:
    with open('models/category_classifier.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None

def calculate_match_score(resume_text, job_description, embedder):
    """Calculates cosine similarity between resume and job description."""
    resume_emb = embedder.encode(resume_text, convert_to_tensor=True)
    jd_emb = embedder.encode(job_description, convert_to_tensor=True)
    similarity = np.dot(resume_emb.cpu(), jd_emb.cpu()) / (np.linalg.norm(resume_emb.cpu()) * np.linalg.norm(jd_emb.cpu()))
    return round(similarity * 100, 2)

def predict_category(resume_fields_text):
    """Predicts job role based on combined resume fields."""
    if model and vectorizer:
        X = vectorizer.transform([resume_fields_text])
        return model.predict(X)[0]
    else:
        return "Unknown"
