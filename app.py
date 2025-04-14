import streamlit as st
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

# Load environment variables
load_dotenv()

# Cache and load the embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Cache and load the Hugging Face LLM model with token
@st.cache_resource(show_spinner=False)
def load_hf_model():
    token = "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"  # your Hugging Face token

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-instruct",
        token=token
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

hf_pipeline = load_hf_model()


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tourism_chatbot")

# Small talk
BASIC_RESPONSES = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there! Ask me anything related to your uploaded documents.",
    "how are you": "I'm just a bot, but I'm here to help! What would you like to ask?",
    "what is your name": "I'm your Tourism Chatbot, here to help with document-based queries!",
}

# -------- File Processing --------
def get_file_text(files):
    text, metadata = "", []
    try:
        for file in files:
            if file.type == "application/pdf":
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    content = page.extract_text()
                    if content:
                        text += content + "\n\n"
                        metadata.append({"document_name": file.name, "page_number": i + 1})
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                for i, para in enumerate(doc.paragraphs):
                    text += para.text + "\n\n"
                    metadata.append({"document_name": file.name, "paragraph_number": i + 1})
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.text(traceback.format_exc())
    return text.strip(), metadata

def get_text_chunks(text, metadata):
    try:
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        chunk_metadata = [metadata[i % len(metadata)] for i in range(len(chunks))]
        return list(zip(chunks, chunk_metadata))
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def generate_embeddings(chunks):
    try:
        return embedding_model.encode([chunk[0] for chunk in chunks], convert_to_numpy=True).tolist()
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

def store_chunks_in_chromadb(chunks):
    try:
        stored_docs = collection.get(include=["documents"]).get("documents", [])
        new_chunks = [chunk for chunk in chunks if chunk[0] not in stored_docs]
        if not new_chunks:
            st.info("No new data to store.")
            return
        embeddings = generate_embeddings(new_chunks)
        for i, (chunk, meta) in enumerate(new_chunks):
            collection.add(
                ids=[f"chunk_{i}_{hash(chunk)}"],
                documents=[chunk],
                embeddings=[embeddings[i]],
                metadatas=[meta]
            )
    except Exception as e:
        st.error(f"Error storing chunks: {e}")
        st.text(traceback.format_exc())

def query_chromadb(query):
    start_time = time.time()
    try:
        if collection.count() == 0:
            return []
        embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[embedding], n_results=5)
        documents, metadatas = results.get("documents", []), results.get("metadatas", [])
        st.write("References:", metadatas)
        duration = time.time() - start_time
        track_query_performance(True, duration, bool(documents))
        return list(zip(documents, metadatas))
    except Exception as e:
        track_query_performance(False, time.time() - start_time, False)
        st.error(f"Error querying ChromaDB: {e}")
        return []

# -------- UI Pages --------
def show_guide_map():
    st.header("Tour Guide Map")
    st.markdown(
        '<iframe src="https://arnob4762.github.io/tour-guide/" width="100%" height="600px" style="border:none;"></iframe>',
        unsafe_allow_html=True
    )

def chatbot_page():
    st.header("Chat with Your Documents")
    user_input = st.text_input("Ask a question about your documents:")

    if user_input:
        query = user_input.lower().strip()
        if query in BASIC_RESPONSES:
            st.write(BASIC_RESPONSES[query])
        else:
            chunks = query_chromadb(user_input)
            context = ""
            references = []
            for doc, meta in chunks:
                ref = meta.get("document_name", "Unknown")
                if "page_number" in meta:
                    ref += f", page {meta['page_number']}"
                if "paragraph_number" in meta:
                    ref += f", paragraph {meta['paragraph_number']}"
                references.append(ref)
                context += f"[{ref}]: {doc}\n\n"

            prompt = (
                f"Context:\n{context}\n\n"
                f"User Question: {user_input}\n\n"
                "Please provide your internal chain-of-thought (prefixed with 'Thinking:') and then your final answer (prefixed with 'Final Answer:')."
            )

            response = hf_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]['generated_text']
            parts = response.split("Final Answer:")
            thinking = parts[0].split("Thinking:")[-1].strip() if "Thinking:" in parts[0] else ""
            final = parts[1].strip() if len(parts) > 1 else response

            if thinking:
                st.markdown(f"<div style='color: gray; font-style: italic;'>&lt;thinking&gt;<br>{thinking}</div>", unsafe_allow_html=True)

            formatted = "<ul>" + "".join([f"<li>{line.strip()}</li>" for line in final.split('\n') if line.strip()]) + "</ul>"
            st.markdown(f"<div style='font-size: 14pt; font-weight: bold;'>{formatted}</div>", unsafe_allow_html=True)

    files = st.file_uploader("Upload your documents (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if files:
        text, meta = get_file_text(files)
        st.write("Extracted Metadata:", meta)
        chunks = get_text_chunks(text, meta)
        store_chunks_in_chromadb(chunks)
        st.success("Documents uploaded and processed successfully!")

def budget_calculator_page():
    st.header("Tour Budget Calculator")
    show_budget_calculator()

# -------- Main App --------
def main():
    st.set_page_config(page_title="Regional Tourism Chatbot", layout="wide")
    st.title("Regional Tourism Chatbot")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Chatbot", "Budget Calculator", "Performance Analyzer", "Guide Map"])

    if choice == "Chatbot":
        chatbot_page()
    elif choice == "Budget Calculator":
        budget_calculator_page()
    elif choice == "Performance Analyzer":
        analyze_performance()
    elif choice == "Guide Map":
        show_guide_map()

if __name__ == "__main__":
    main()
