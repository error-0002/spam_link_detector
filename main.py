import os
import sys
import time
from train_model import train_model  # your training script

if __name__ == "__main__":
    # 1️⃣ Train model
    print("Starting model training...")
    model_path = train_model()
    print(f"✅ Model trained and saved at {model_path}")

    # 2️⃣ Start Flask app in background
    print("\n🚀 Starting Flask backend...")
    os.system(f'start cmd /k "{sys.executable} app.py"')
    time.sleep(3)  # wait 3 sec for server to start

    # 3️⃣ Launch Streamlit UI automatically
    print("\n🌐 Launching Streamlit web app...")
    os.system(f'{sys.executable} -m streamlit run ui_app.py')
