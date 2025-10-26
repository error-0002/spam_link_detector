# ui_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Spam Message Detector")

st.title("📩 Spam Message & Link Detector")
st.write("Paste your message or SMS below 👇")

msg = st.text_area("Message:", height=200)

# --- URL of your deployed Flask backend ---
BACKEND_URL = "https://spam-detector-fgop.onrender.com/predict"

if st.button("Check Message"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        try:
            res = requests.post(BACKEND_URL, json={"message": msg})
            data = res.json()

            if "error" in data:
                st.error(f"Server error: {data['error']}")
            else:
                st.subheader(f"Prediction: {data['prediction']}")
                st.write(f"Confidence: {data['confidence']:.2f}")

                # Optional: show link info if backend sends it
                if data.get("has_link"):
                    st.info("🔗 Link detected in message")
                else:
                    st.success("No link detected.")
        except Exception as e:
            st.error(f"Error: {e}")
