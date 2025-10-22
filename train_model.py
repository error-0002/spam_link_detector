# train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_model():
    os.makedirs("model", exist_ok=True)
    
    # Load your dataset
    df = pd.read_csv("data/sms_spam.tsv", sep="\t", header=None, names=["label", "message"])
    df = df.drop_duplicates(subset="message")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)
    
    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("model", MultinomialNB())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {acc:.2f}")
    
    # Save pipeline
    model_path = "model/spam_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"💾 Saved pipeline to {model_path}")
    
    return model_path  # important — main.py expects this

# Optional: If you run this file alone
if __name__ == "__main__":
    train_model()
