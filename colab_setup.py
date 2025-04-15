import os
import subprocess
import time

# Step 1: Install required packages
print("🔧 Installing requirements from requirements.txt...")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# (Gradio is already in requirements.txt — no need to install separately)
# So Step 2 is not needed.

# Step 2: Launch Gradio app
print("🚀 Launching Gradio app...")
import app  # Make sure app.py contains `demo = gr.Interface(...)`

# Launch the app with shareable public link (good for Colab)
app.demo.launch(share=True)

# Step 3: Prevent Colab from disconnecting
print("⏳ Keeping Colab session alive...")
while True:
    time.sleep(60)
