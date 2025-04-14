# **🌍 Travelmate BD – Regional Tourism Chatbot**

**Regional Tourism Chatbot** is an AI-powered chatbot designed to assist users in planning and exploring regional tours across Bangladesh. Developed as part of the "CSE 299 – Junior Design Project (Spring 2025, Section 19)" at North South University, this web-based application leverages natural language processing to deliver destination guidance, budget estimation, and document-based question answering — all through a user-friendly interface built with **Streamlit** and deployed via **Google Colab** using **ngrok**.

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

- **AI Chatbot**: Travel-focused Q&A using `deepseek-ai/deepseek-llm-7b-base` from Hugging Face.
- **Document QA**: Context-aware answers from uploaded PDF or DOCX files.
- **Tour Budget Calculator**: Estimates costs for transportation, accommodation, and more.
- **Performance Analyzer**: Monitors memory and response time in real time.
- **Web-Based Interface**: Runs seamlessly on Google Colab with public access via ngrok.

---

## 🚀 How to Run (Google Colab Only)

1. Open a new Google Colab notebook.
2. Clone the repository:
   ```python
   !git clone https://github.com/Arnob4762/CSE299-PROJECT-Travelmate-BD---Regional-Tourism-Chatbot-
   %cd CSE299-PROJECT-Travelmate-BD---Regional-Tourism-Chatbot-
   ```
3. Run the setup script:
   ```python
   !python colab_setup.py
   ```
4. After initialization, an **ngrok URL** will be displayed. Click the URL to access the chatbot in your browser.

---

## 🛠 Technologies Used

- Python  
- Streamlit  
- Hugging Face Transformers  
- ChromaDB  
- Google Colab  
- Ngrok  
- PyPDF2, python-docx  
- psutil

---

## 📄 License & Acknowledgements

This project is intended for educational purposes only.  
We gratefully acknowledge the guidance of **Dr. Shafin Rahman** 

---

```
