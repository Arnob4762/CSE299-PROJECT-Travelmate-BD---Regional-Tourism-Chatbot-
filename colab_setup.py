import os
import subprocess
import time

# Step 1: Clone your GitHub repo (optional if already uploaded)
# subprocess.run(["git", "clone", "https://github.com/your-username/your-repository-name.git"])

# Step 2: Install required packages
print("Installing requirements...")
subprocess.run(["pip", "install", "-r", "requirement.txt"])

# Step 3: Install and configure ngrok
print("Setting up ngrok...")
subprocess.run(["pip", "install", "pyngrok"])
from pyngrok import ngrok

# Optional: Add your authtoken if you have one
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")  # Or paste token here as string
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Step 4: Start Streamlit app with ngrok tunnel
print("Launching Streamlit app with ngrok tunnel...")
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")

# Step 5: Run Streamlit app
subprocess.Popen(["streamlit", "run", "app.py"])

# Keep the notebook alive to maintain the tunnel
while True:
    time.sleep(60)
