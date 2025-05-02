import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Training Data
train_data = pd.read_csv('data/resume_data/Train_Resume_Data.csv')
train_data.columns = train_data.columns.str.strip()

# 2. Combine useful fields into a single text
train_data['combined_text'] = (
    train_data['Skills'].fillna('') + ' ' +
    train_data['Experience (Years)'].astype(str).fillna('') + ' ' +
    train_data['Education'].fillna('') + ' ' +
    train_data['Certifications'].fillna('') + ' ' +
    train_data['Job Role'].fillna('')
)

# 3. Preprocessing
X_train = train_data['combined_text']
y_train = train_data['Job Role']  # or 'Recruiter Decision' if you want to predict selection instead

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)

# 5. Model Training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vect, y_train)

# 6. Save Model + Vectorizer
with open('models/category_classifier.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model and vectorizer saved to models/category_classifier.pkl")

# ------------------------------
# (Optional) Evaluate on Test Data
# ------------------------------
try:
    test_data = pd.read_csv('data/resume_data/Test_Resume_Data.csv')
    test_data.columns = test_data.columns.str.strip()

    test_data['combined_text'] = (
        test_data['Skills'].fillna('') + ' ' +
        test_data['Experience (Years)'].astype(str).fillna('') + ' ' +
        test_data['Education'].fillna('') + ' ' +
        test_data['Certifications'].fillna('') + ' ' +
        test_data['Job Role'].fillna('')
    )

    X_test = test_data['combined_text']
    y_test = test_data['Job Role']
    X_test_vect = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📈 Test Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    print(f"⚠️ Could not evaluate on test set: {e}")
