import os
import subprocess
import time

# Step 1: Install required packages
print("Installing requirements...")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Step 2: Install Gradio
print("Setting up Gradio...")
subprocess.run(["pip", "install", "gradio"])

# Step 3: Launch Gradio app
print("Launching Gradio app...")

# Ensure app.py is the name of your Gradio script, or adjust accordingly
import app  # Assuming your Gradio app is in a file named `app.py`

# This will launch the app with a public URL
app.demo.launch(share=True)

# Step 4: Keep the app running (Colab may stop execution otherwise)
while True:
    time.sleep(60)



