import gradio as gr
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import traceback
from tour_budget import show_budget_calculator
from performance_analyzer import track_query_performance, analyze_performance
import os

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load DeepSeek R1 Distilled Qwen 1.5B model pipeline
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

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tourism_chatbot")

# Small talk predefined responses
BASIC_RESPONSES = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there! Ask me anything related to your uploaded documents.",
    "how are you": "I'm just a bot, but I'm here to help! What would you like to ask?",
    "what is your name": "I'm your Tourism Chatbot, here to help with document-based queries!",
}

# File processing

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
                        metadata.append({"document_name": file.name, "page_number": i + 1})
            elif file.name.endswith(".docx"):
                doc = docx.Document(file)
                for i, para in enumerate(doc.paragraphs):
                    text += para.text + "\n\n"
                    metadata.append({"document_name": file.name, "paragraph_number": i + 1})
    except Exception as e:
        print(f"Error reading files: {e}")
        print(traceback.format_exc())

    # Debugging: print the extracted text to verify it's correct
    print(f"Extracted text:\n{text[:500]}...")  # Print first 500 characters for debugging
    return text.strip(), metadata

def get_text_chunks(text, metadata):
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    chunk_metadata = [metadata[i % len(metadata)] for i in range(len(chunks))]

    # Debugging: print out the first few chunks to verify chunking
    print(f"Chunks:\n{chunks[:3]}")  # Print first 3 chunks for debugging
    return list(zip(chunks, chunk_metadata))

def generate_embeddings(chunks):
    return embedding_model.encode([chunk[0] for chunk in chunks], convert_to_numpy=True).tolist()

def store_chunks_in_chromadb(chunks):
    stored_docs = collection.get(include=["documents"]).get("documents", [])
    new_chunks = [chunk for chunk in chunks if chunk[0] not in stored_docs]
    if not new_chunks:
        return "No new data to store."
    embeddings = generate_embeddings(new_chunks)
    for i, (chunk, meta) in enumerate(new_chunks):
        collection.add(
            ids=[f"chunk_{i}_{hash(chunk)}"],
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[meta]
        )
    return "Documents uploaded and processed successfully!"

def query_chromadb(query):
    start_time = time.time()
    try:
        if collection.count() == 0:
            return []
        embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[embedding], n_results=5)
        documents, metadatas = results.get("documents", []), results.get("metadatas", [])
        duration = time.time() - start_time
        track_query_performance(True, duration, bool(documents))
        return list(zip(documents, metadatas))
    except Exception as e:
        track_query_performance(False, time.time() - start_time, False)
        print(f"Error querying ChromaDB: {e}")
        return []

# Gradio tab: Chatbot

def chat_with_documents(user_input, files):
    if user_input.lower().strip() in BASIC_RESPONSES:
        return BASIC_RESPONSES[user_input.lower().strip()]

    if files:
        text, meta = get_file_text(files)
        chunks = get_text_chunks(text, meta)
        store_chunks_in_chromadb(chunks)

    chunks = query_chromadb(user_input)
    context = ""
    references = []

    for doc, meta in chunks:
        refs = []
        if isinstance(meta, list):
            for m in meta:
                ref = m.get("document_name", "Unknown")
                if "page_number" in m:
                    ref += f", page {m['page_number']}"
                if "paragraph_number" in m:
                    ref += f", paragraph {m['paragraph_number']}"
                refs.append(ref)
        else:
            ref = meta.get("document_name", "Unknown")
            if "page_number" in meta:
                ref += f", page {meta['page_number']}"
            if "paragraph_number" in meta:
                ref += f", paragraph {meta['paragraph_number']}"
            refs.append(ref)

        references.extend(refs)
        for r in refs:
            context += f"[{r}]: {doc}\n\n"

    prompt = (
        f"Context:\n{context}\n\n"
        f"User Question: {user_input}\n\n"
        "Please think carefully before responding. Your final answer should be helpful and grounded in the provided context."
    )

    response = hf_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]
    response_text = response.get("generated_text", "[No response]") if isinstance(response, dict) else response

    references_markdown = "**References:**\n" + "\n".join(f"- {r}" for r in references) if references else "*No references found.*"
    
    # Return only the final answer without showing reasoning
    return f"**Response:**\n{response_text}\n\n{references_markdown}"

# Gradio tab: Budget Calculator

def budget_tab():
    return show_budget_calculator()

# Gradio tab: Performance Analyzer

def performance_tab():
    return analyze_performance()

# Gradio tab: Guide Map

def guide_map_tab():
    return '<iframe src="https://arnob4762.github.io/tour-guide/" width="100%" height="600px" style="border:none;"></iframe>'

# Gradio UI setup
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
        guide_map_iframe = gr.HTML(guide_map_tab())

demo.launch(share=True)
