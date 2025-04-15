
# 🌍 Travelmate BD – Regional Tourism Chatbot

**Travelmate BD** is an AI-powered tourism assistant designed to help users explore and plan trips across **Bangladesh**. Built with **Gradio**, it features a multilingual chatbot, tour budget calculator, document-based Q&A, and real-time performance analyzer — all optimized to run seamlessly on **Google Colab**.

Developed as part of the **CSE 299 – Junior Design Project (Spring 2025, Section 19)** at **North South University**, this tool combines user-friendly interfaces with powerful NLP models.

---

## 📌 Project Overview

- **Project Title**: Travelmate BD – Regional Tourism Chatbot  
- **Course**: CSE 299 – Junior Design Project (Spring 2025)  
- **Section**: 19  
- **Supervisor**: Dr. Shafin Rahman  
- **Group Number**: 2  

**Team Members**:
- Azmain Iqtidar Arnob  
- Md Nayeem Porag Molla  
- Atikul Islam Nahid  
- Md Ashraful Islam

---

## ⚙️ Key Features

- 🤖 **AI Chatbot**  
  Ask travel-related questions in Bangla or English — powered by [`deepseek-ai/deepseek-llm-7b-base`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base).

- 📄 **Document QA**  
  Upload `.pdf` or `.docx` files and receive context-aware answers extracted from the content.

- 💸 **Tour Budget Calculator**  
  Plan trips by estimating transport, accommodation, and food costs based on your preferences.

- 📊 **Performance Analyzer**  
  Monitor memory usage and response latency during chatbot interaction.

- 🌐 **Web-Based Interface**  
  Powered by Gradio and deployable via a shareable link on **Google Colab**.

---

## 🚀 How to Run (Google Colab Only)

1. **Open a new Colab notebook**.
2. **Clone this repository**:
   ```bash
   !git clone https://github.com/Arnob4762/CSE299-PROJECT-Travelmate-BD---Regional-Tourism-Chatbot-
   %cd CSE299-PROJECT-Travelmate-BD---Regional-Tourism-Chatbot-
   ```
3. **Run the setup script**:
   ```bash
   !python colab_setup.py
   ```
4. After setup, a **Gradio public link** will appear — click it to access the full app in your browser.

---

## 🧠 Architecture Overview

The application has four main interfaces (Gradio tabs):
1. **Chatbot** – Chat with the AI about destinations, weather, travel advice, and more.
2. **Budget Calculator** – Select destination, hotel, transport, and restaurant options to generate an expense summary.
3. **Performance Analyzer** – Tracks memory usage and response latency using `psutil`.
4. **Guide Map (Coming Soon)** – Interactive map-based exploration (currently in development).

All components share **persistent state**, allowing seamless switching between tabs without losing context.

---

## 🛠 Tech Stack

- **Python**  
- **Gradio**  
- **Hugging Face Transformers**  
- **ChromaDB** (for document embeddings)  
- **Google Colab**  
- **sentence-transformers**, `faiss-cpu`  
- **PyPDF2**, `python-docx`, `psutil`  
- **torch`, `accelerate`, `numpy`, `scipy`  

---

## 📄 License & Acknowledgements

This project is created for educational purposes under the supervision of **Dr. Shafin Rahman**.  
All third-party models and datasets are used in accordance with their respective licenses.
