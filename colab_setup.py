import os
import subprocess
import time
import sys

# Step 1: Install required packages
print("🔧 Installing requirements from requirements.txt...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Step 2: Launch Gradio app
print("🚀 Launching Gradio app...")

try:
    import app  # Ensure app.py defines demo = gr.Blocks(...) or similar
    app.demo.launch(share=True)
except Exception as e:
    print("❌ Failed to launch app. Error:")
    print(e)
    sys.exit(1)

# Step 3: Prevent Colab from disconnecting
print("⏳ Keeping Colab session alive (useful for long sessions)...")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("🛑 Session ended manually.")
