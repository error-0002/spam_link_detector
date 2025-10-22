# ui_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Spam Message Detector")

st.title("📩 Spam Message & Link Detector")
st.write("Paste your message or SMS below 👇")

msg = st.text_area("Message:", height=200)

if st.button("Check Message"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        try:
            res = requests.post("http://127.0.0.1:5000/predict", json={"message": msg})
            data = res.json()
            st.subheader(f"Prediction: {data['prediction']}")
            st.write(f"Confidence: {data['confidence']:.2f}")
            if data['detected_links']:
                st.write("🔗 Links Found:")
                for u in data['detected_links']:
                    st.write("-", u)
            else:
                st.write("No links detected.")
        except Exception as e:
            st.error(f"Error: {e}")
