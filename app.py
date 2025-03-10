import re
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
from performance_analyzer import track_query_performance, analyze_performance
from tour_budget import show_budget_calculator  # Budget Estimation page

# Load environment variables
load_dotenv()

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tourism_chatbot")

# Predefined responses for small talk
BASIC_RESPONSES = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there! Ask me anything related to your uploaded documents.",
    "how are you": "I'm just a bot, but I'm here to help! What would you like to ask?",
    "what is your name": "I'm your Tourism Chatbot, here to help with document-based queries!",
}

# ----- DOCUMENT PROCESSING FUNCTIONS -----
def get_file_text(files):
    text = ""
    metadata = []
    try:
        for file in files:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        metadata.append({
                            "document_name": file.name,
                            "page_number": page_num + 1
                        })
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                for para_num, para in enumerate(doc.paragraphs):
                    text += para.text + "\n\n"
                    metadata.append({
                        "document_name": file.name,
                        "paragraph_number": para_num + 1
                    })
    except Exception as e:
        st.error(f"Error reading files: {e}")
    return text.strip(), metadata

def get_text_chunks(text, metadata):
    try:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_text(text)
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            meta = metadata[i % len(metadata)] if metadata else {}
            chunk_metadata.append((chunk, meta))
        return chunk_metadata
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def generate_embeddings(chunks):
    try:
        texts = [chunk[0] for chunk in chunks]
        return embedding_model.encode(texts, convert_to_numpy=True).tolist()
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
                metadatas=[meta]  # Storing metadata correctly
            )
    except Exception as e:
        st.error(f"Error storing chunks in ChromaDB: {e}")

def query_chromadb(query):
    try:
        if collection.count() == 0:
            return []
        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        # Extract document text and metadata correctly
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        return [(doc, meta) for doc, meta in zip(docs, metas)]
    except Exception as e:
        st.error(f"Error querying ChromaDB: {e}")
        return []

# ----- CHATBOT FUNCTIONS -----
def query_ollama(prompt):
    start_time = time.time()
    success = False
    error_occurred = False
    try:
        response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
        success = True
        track_query_performance(start_time, success, error_occurred)
        reply = response["message"]["content"]
        # Force response in English
        reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
        return reply
    except Exception as e:
        error_occurred = True
        track_query_performance(start_time, success, error_occurred)
        st.error(f"Error querying LLM: {e}")
        return "I'm sorry, but I couldn't generate a response."

def chatbot_page():
    st.header("Regional Tourism Chatbot")
    st.markdown("[🌍 Click to View Regional Tourism Guide Map](https://arnob4762.github.io/tour-guide/)")
    user_input = st.text_input("Ask a question about your documents:")
    
    if user_input:
        lower_input = user_input.lower().strip()
        if lower_input in BASIC_RESPONSES:
            st.write(BASIC_RESPONSES[lower_input])
        else:
            relevant_chunks = query_chromadb(user_input)
            if relevant_chunks:
                context = "\n".join([str(doc) for doc, _ in relevant_chunks])
            else:
                st.warning("No relevant data found. The chatbot may give a generic response.")
                context = "No relevant data found."
            
            prompt = f"Context:\n{context}\n\nUser Question (in English only): {user_input}\nAnswer in English only:"
            response = query_ollama(prompt)
            
            if "no relevant data found" in context.lower():
                response = "Sorry, I can only assist with information related to your uploaded documents."
            st.write(response)
            
            # Display document references at the end, if available
            if relevant_chunks:
                st.markdown("#### Document References")
                for doc, meta in relevant_chunks:
                    if isinstance(meta, dict):
                        doc_name = meta.get("document_name", None)
                        page = meta.get("page_number", None)
                        paragraph = meta.get("paragraph_number", None)
                        
                        ref_title = f"Document: {doc_name or 'Unknown'}"
                        if page:
                            ref_title += f" | Page: {page}"
                        if paragraph:
                            ref_title += f" | Paragraph: {paragraph}"
                        
                        with st.expander(ref_title):
                            st.write(doc)

    st.subheader("Upload Your Documents")
    uploaded_files = st.file_uploader("Upload PDFs & DOCs", accept_multiple_files=True)
    
    if st.button("Process") and uploaded_files:
        raw_text, metadata = get_file_text(uploaded_files)
        if raw_text:
            text_chunks = get_text_chunks(raw_text, metadata)
            store_chunks_in_chromadb(text_chunks)
            st.success("✅ Processing complete! Documents are now searchable.")

# ----- PERFORMANCE ANALYSIS PAGE -----
def performance_page():
    st.header("Performance Analysis")
    st.markdown("Below are the performance metrics of your chatbot:")
    analyze_performance()

# ----- MAIN APP -----
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chatbot", "Budget Estimation", "Performance Analysis"])
    
    if page == "Chatbot":
        chatbot_page()
    elif page == "Budget Estimation":
        show_budget_calculator()
    elif page == "Performance Analysis":
        performance_page()

if __name__ == "__main__":
    main()
