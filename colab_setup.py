import os
import subprocess
import time

# Step 1: Install required packages
print("Installing requirements...")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Step 2: Install and configure ngrok
print("Setting up ngrok...")
subprocess.run(["pip", "install", "pyngrok"])
from pyngrok import ngrok

# Set your ngrok auth token directly here
NGROK_AUTH_TOKEN = "2u8OrgSVoCMb3qCS2D6aaKjrRZf_63NcpbjYCzguKgDHqS8Ys"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Step 3: Start Streamlit app with ngrok tunnel
print("Launching Streamlit app with ngrok tunnel...")
public_url = ngrok.connect(8501)
print(f"🌐 Your app is live at: {public_url}")

# Step 4: Run Streamlit app
subprocess.Popen(["streamlit", "run", "app.py"])

# Step 5: Keep the tunnel alive
while True:
    time.sleep(60)

