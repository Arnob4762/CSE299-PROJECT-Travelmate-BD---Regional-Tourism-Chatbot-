# Travelmate BD – Regional Tourism Chatbot

Welcome to **Travelmate BD**, a smart regional tourism chatbot designed as part of the CSE299 Junior Design Project at BRAC University.

## 📌 Project Overview
This chatbot assists users in planning their trips by:
- Providing information about Bangladeshi tourist destinations
- Answering questions using uploaded documents
- Calculating personalized travel budgets
- Displaying helpful tools like maps and performance insights

Built using:
- 🧠 DeepSeek R1 7B model via Hugging Face
- 🧮 LangChain + ChromaDB for local document QA
- 🧾 Streamlit for user interface
- 📊 Budget & performance calculators

---

## 🚀 Features
- 🧳 **Tour Budget Calculator** – Get estimated costs based on transport, hotel, food, and trip length.
- 🧠 **Chatbot (LLM)** – Powered by DeepSeek R1 7B, gives human-like responses.
- 📑 **Document QA** – Upload a PDF/DOCX and ask questions from it.
- 📈 **Performance Analyzer** – Measures response accuracy and latency.
- 🗺️ **Tour Map** – Visual map of popular destinations (via iframe).

---

## ⚙️ Installation (Locally or Colab)

### Option 1: Google Colab + Ngrok (Recommended for easy hosting)
1. Open the repo in Colab.
2. Run this setup:
   ```python
   !pip install -r requirements.txt
   !pip install pyngrok
   from pyngrok import ngrok
   public_url = ngrok.connect(8501)
   !streamlit run app.py & npx localtunnel --port 8501
   print(f"App running at: {public_url}")
