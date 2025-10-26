import os
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Load Model ---
MODEL_PATH = "model/spam_pipeline.joblib"

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    pipeline = None
    print(f"⚠️ Warning: Failed to load model from {MODEL_PATH}: {e}")


def has_link(text):
    """Detect if message contains a link"""
    return bool(re.search(r"http[s]?://|www\.|link", text.lower()))


def predict_spam(message):
    """Predict spam or ham"""
    if not pipeline:
        return {"prediction": "UNKNOWN", "confidence": 0.0, "has_link": has_link(message)}

    prediction = pipeline.predict([message])[0]
    prob = pipeline.predict_proba([message])[0].max()

    msg_lower = message.lower()
    spam_keywords = [
        "win", "free", "prize", "gift", "reward", "claim", "click", "offer",
        "congratulations", "urgent", "selected", "lottery", "bonus", "discount"
    ]
    contains_link = has_link(msg_lower)

    if contains_link or any(word in msg_lower for word in spam_keywords):
        if prediction == "ham" and prob < 0.95:
            prediction = "spam"
            prob = max(prob, 0.99)

    return {
        "prediction": prediction.upper(),
        "confidence": round(float(prob), 2),
        "has_link": contains_link
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Spam Detector API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for spam prediction"""
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        if not message.strip():
            return jsonify({"error": "Empty message"}), 400

        result = predict_spam(message)
        return jsonify(result)
    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
