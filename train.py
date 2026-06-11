import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (only required columns)
df = pd.read_csv(r"news.csv", usecols=["text", "label"], low_memory=False)

# Drop rows where text or label is missing
df = df.dropna(subset=["text", "label"])

# Convert text to string
df["text"] = df["text"].astype(str)

# Map labels: real → 0, fake → 1 (if needed)
if df["label"].dtype == object:   # only map if text labels
    df["label"] = df["label"].map({"real": 0, "fake": 1})

# Separate features and target
X = df["text"]
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")
