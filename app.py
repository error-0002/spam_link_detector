import os
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)

# --- Load trained model ---
MODEL_PATH = "model/spam_pipeline.joblib"

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    pipeline = None
    print(f"⚠️ Warning: Failed to load model from {MODEL_PATH}: {e}")


# --- Helper function to detect links ---
def has_link(text):
    return bool(re.search(r"http[s]?://|www\\.|link", text.lower()))


# --- Helper function to predict spam ---
def predict_spam(message):
    if not pipeline:
        # fallback if model not loaded
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "has_link": has_link(message),
        }

    # ML model prediction
    prediction = pipeline.predict([message])[0]
    prob = pipeline.predict_proba([message])[0].max()

    # --- Rule-based boost ---
    msg_lower = message.lower()
    spam_keywords = [
        "win", "free", "prize", "gift card", "cash", "reward", "claim",
        "click", "offer", "congratulations", "urgent", "selected",
        "lottery", "bonus", "discount"
    ]

    contains_link = has_link(msg_lower)

    # If message looks spammy but ML missed it, override
    if contains_link or any(word in msg_lower for word in spam_keywords):
        if prediction == "ham" and prob < 0.95:
            prediction = "spam"
            prob = max(prob, 0.99)

    return {
        "prediction": prediction.upper(),
        "confidence": round(float(prob), 2),
        "has_link": contains_link
    }


# --- API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    result = predict_spam(message)
    return jsonify(result)


# --- Health Check Route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "📩 Spam Detector API is running!"})


# --- Run Flask Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
