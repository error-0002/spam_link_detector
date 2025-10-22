# test_predict.py
import joblib
pipeline = joblib.load("model/spam_pipeline.joblib")
samples = [
    "Congratulations! You have won free tickets. Click link to claim now.",
    "Hey bro, are we meeting at 6pm?"
]
for s in samples:
    pred = pipeline.predict([s])[0]
    prob = pipeline.predict_proba([s])[0]
    print(s, "->", "SPAM" if pred==1 else "HAM", "prob:", max(prob))
