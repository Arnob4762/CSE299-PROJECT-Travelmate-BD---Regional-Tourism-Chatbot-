import gradio as gr
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import traceback
import numpy as np
import faiss
import pickle
from tour_budget import show_budget_calculator
from performance_analyzer import track_query_performance, analyze_performance

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load DeepSeek model pipeline
token = os.environ.get("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    token=token,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    token=token,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# File handling functions
def get_file_text(files):
    text, metadata = "", []
    try:
        for file in files:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    content = page.extract_text()
                    if content:
                        text += content + "\n\n"
                        metadata.append((file.name, f"page {i + 1}"))
            elif file.name.endswith(".docx"):
                doc = docx.Document(file)
                for i, para in enumerate(doc.paragraphs):
                    text += para.text + "\n\n"
                    metadata.append((file.name, f"paragraph {i + 1}"))
    except Exception as e:
        print("Error reading files:", e)
        print(traceback.format_exc())
    return text.strip(), metadata

# Split and embed
text_chunks, meta_chunks = [], []
faiss_index = None

def process_and_store_chunks(text, metadata):
    global faiss_index, text_chunks, meta_chunks
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    chunk_meta = [metadata[i % len(metadata)] for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    text_chunks = chunks
    meta_chunks = chunk_meta
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))

# Retrieve
def retrieve_context(query, k=5):
    if not faiss_index or not text_chunks:
        return []
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, k)
    results = [(text_chunks[i], meta_chunks[i]) for i in I[0]]
    return results

# Small talk
BASIC_RESPONSES = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there! Ask me anything related to your uploaded documents.",
    "how are you": "I'm just a bot, but I'm here to help!",
    "what is your name": "I'm your Tourism Chatbot!",
}

# Chatbot response
def chat_with_documents(user_input, files):
    if user_input.lower().strip() in BASIC_RESPONSES:
        return BASIC_RESPONSES[user_input.lower().strip()]

    if files:
        text, meta = get_file_text(files)
        process_and_store_chunks(text, meta)

    results = retrieve_context(user_input)
    context = "\n".join([f"[{m[0]}, {m[1]}]: {c}" for c, m in results])

    prompt = (
        f"Context:\n{context}\n\n"
        f"User Question: {user_input}\n\n"
        f"Just provide a clear and concise answer based only on the context. Avoid extra reasoning or justification."
    )

    response = hf_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]
    text_only = response["generated_text"] if isinstance(response, dict) else response
    return f"**Response:**\n{text_only}"

# Performance analyzer tracking
def track_query_performance(user_input, files):
    return analyze_performance(user_input, files)

# Gradio tabs
def budget_tab():
    return show_budget_calculator()

def performance_tab():
    with gr.Column():
        input_box = gr.Textbox(label="Enter a query to analyze:")
        file_input = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"], file_count="multiple")
        output_box = gr.Markdown()
        analyze_button = gr.Button("Analyze Performance")
        analyze_button.click(fn=analyze_performance, inputs=[input_box, file_input], outputs=output_box)


def guide_map_tab():
    return '<iframe src="https://arnob4762.github.io/tour-guide/" width="100%" height="600px" style="border:none;"></iframe>'

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧭 Regional Tourism Chatbot")

    with gr.Tab("📄 Chatbot"):
        chatbot_input = gr.Textbox(label="Ask a question about your documents:")
        chatbot_files = gr.File(label="Upload PDF or DOCX", file_types=['.pdf', '.docx'], file_count="multiple")
        chatbot_output = gr.Markdown()
        chatbot_button = gr.Button("Get Answer")
        chatbot_button.click(fn=chat_with_documents, inputs=[chatbot_input, chatbot_files], outputs=chatbot_output)

    with gr.Tab("💰 Budget Calculator"):
        budget_tab()

    with gr.Tab("📊 Performance Analyzer"):
        performance_tab()

    with gr.Tab("🗺️ Guide Map"):
        gr.HTML(guide_map_tab())

if __name__ == "__main__":
    demo.launch(share=True)
