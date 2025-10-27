import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# 1️⃣ Load your dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# 2️⃣ Keep only relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 3️⃣ Clean the text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove strange chars
    text = re.sub(r'http\S+', '', text)         # remove links
    text = re.sub(r'\d+', '', text)             # remove numbers
    text = text.lower().strip()
    return text

df['message'] = df['message'].apply(clean_text)

# 4️⃣ Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 5️⃣ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6️⃣ Train Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 7️⃣ Evaluate model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 8️⃣ Save model & vectorizer
joblib.dump(model, "model/spam_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("✅ Model and vectorizer saved successfully!")
