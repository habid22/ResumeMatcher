import spacy
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained classifier
try:
    with open('models/category_classifier.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None

# Sentence-BERT match score
def calculate_match_score(resume_text, job_description, embedder):
    resume_emb = embedder.encode(resume_text, convert_to_tensor=True)
    jd_emb = embedder.encode(job_description, convert_to_tensor=True)
    similarity = np.dot(resume_emb.cpu(), jd_emb.cpu()) / (np.linalg.norm(resume_emb.cpu()) * np.linalg.norm(jd_emb.cpu()))
    return round(similarity * 100, 2)

# Predict resume category
def predict_category(resume_text):
    if model and vectorizer:
        X = vectorizer.transform([resume_text])
        return model.predict(X)[0]
    else:
        return "Unknown"

# --- 🔥 Dynamic Skill Extraction Only ---

def extract_entities(text):
    """Use spaCy NER to extract PRODUCT, ORG, MISC entities from text."""
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "MISC", "WORK_OF_ART"]:
            clean_ent = ent.text.strip().lower()
            if len(clean_ent) > 2 and clean_ent not in stop_words:
                entities.add(clean_ent)
    return entities

def suggest_resume_improvements_dynamic(job_description, resume_text):
    """Compare dynamic extracted entities from JD and Resume, suggest missing."""
    jd_entities = extract_entities(job_description)
    resume_entities = extract_entities(resume_text)

    missing_entities = jd_entities - resume_entities

    return sorted(missing_entities)
