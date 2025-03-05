import streamlit as st  # UI
from dotenv import load_dotenv  # Load environment variables
from PyPDF2 import PdfReader  # Read PDFs
import docx  # Read DOCX files
from langchain.text_splitter import CharacterTextSplitter  # Text chunking
import chromadb  # Vector database for retrieval
from sentence_transformers import SentenceTransformer  # Embeddings
import numpy as np  # Handle embedding arrays
import ollama  # LLM for answering queries
import time  # For performance timing
from performance_analyzer import track_query_performance, analyze_performance  # Import performance functions

# Load environment variables
load_dotenv()

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tourism_chatbot")

# Predefined responses for small talk
BASIC_RESPONSES = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there! Ask me anything related to your uploaded documents.",
    "how are you": "I'm just a bot, but I'm here to help! What would you like to ask?",
    "what is your name": "I'm your Tourism Chatbot, here to help with document-based queries!",
}

# Confidentiality filter for restricted topics
CONFIDENTIAL_QUESTIONS = [
    "what model are you using",
    "what ai are you using",
    "which llm is this",
    "are you using deepseek",
    "are you using gpt",
    "are you based on openai",
    "who built you",
    "how were you made",
    "who created you",
    "tell me your code",
    "show me your code",
    "which framework are you using",
]

# Extract text from PDFs & DOCXs
def get_file_text(files):
    text = ""
    try:
        for file in files:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n\n"
    except Exception as e:
        st.error(f"Error reading files: {e}")
    return text.strip()

# Split text into manageable chunks
def get_text_chunks(text):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,  
            chunk_overlap=100,
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Generate embeddings for text chunks
def generate_embeddings(chunks):
    try:
        return embedding_model.encode(chunks, convert_to_numpy=True).tolist()
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Store chunks in ChromaDB
def store_chunks_in_chromadb(chunks):
    try:
        stored_docs = collection.get(include=["documents"]).get("documents", [])
        new_chunks = [chunk for chunk in chunks if chunk not in stored_docs]

        if not new_chunks:
            st.info("No new data to store.")
            return

        embeddings = generate_embeddings(new_chunks)
        for i, (chunk, embedding) in enumerate(zip(new_chunks, embeddings)):
            collection.add(
                ids=[f"chunk_{i}_{hash(chunk)}"],  # Unique ID
                documents=[chunk],  # Store text
                embeddings=[embedding]  # Store vector
            )
    except Exception as e:
        st.error(f"Error storing chunks in ChromaDB: {e}")

# Retrieve relevant text chunks from ChromaDB
def query_chromadb(query):
    try:
        if collection.count() == 0:
            return []  # Prevent errors if database is empty

        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        return results.get("documents", [[]])[0]
    except Exception as e:
        st.error(f"Error querying ChromaDB: {e}")
        return []

# Query DeepSeek R1 for responses
def query_ollama(prompt):
    start_time = time.time()  # Start timing the response
    success = False
    error_occurred = False

    try:
        response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
        success = True
        response_time = time.time() - start_time  # Calculate response time
        track_query_performance(start_time, success, error_occurred)  # Track performance data
        return response["message"]["content"]
    except Exception as e:
        error_occurred = True
        track_query_performance(start_time, success, error_occurred)  # Track performance data
        st.error(f"Error querying LLM: {e}")
        return "I'm sorry, but I couldn't generate a response."

# Main Streamlit App
def main():
    st.set_page_config(page_title="Regional Tourism Chatbot", page_icon="🌍")

    st.header("REGIONAL TOURISM CHATBOT ")
    st.markdown("[🌍 Click to View Regional Tourism Guide Map](https://arnob4762.github.io/tour-guide/)")

    user_input = st.text_input("Ask a question about your documents:")

    if user_input:
        lower_input = user_input.lower().strip()

        # Confidentiality filter
        if any(q in lower_input for q in CONFIDENTIAL_QUESTIONS):
            st.write("Sorry, that's confidential. I cannot disclose that.")
            return
        
        # Check for basic conversational responses
        if lower_input in BASIC_RESPONSES:
            st.write(BASIC_RESPONSES[lower_input])
        else:
            relevant_chunks = query_chromadb(user_input)
            
            if relevant_chunks:
                context = "\n".join(relevant_chunks)
            else:
                st.warning("No relevant data found. The chatbot may give a generic response.")
                context = "No relevant data found."

            prompt = f"Context:\n{context}\n\nUser Question: {user_input}"
            response = query_ollama(prompt)

            # If response is too generic, give a fallback response
            if "no relevant data found" in context.lower():
                response = "Sorry, I can only assist with information related to your uploaded documents."

            st.write(response)

    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        uploaded_files = st.file_uploader("Upload PDFs & DOCs", accept_multiple_files=True)

        if st.button("Process") and uploaded_files:
            raw_text = get_file_text(uploaded_files)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                store_chunks_in_chromadb(text_chunks)
                st.success("✅ Processing complete! Documents are now searchable.")

        st.subheader("🔍 Performance Analyzer")
        if st.button("Analyze Performance"):
            analyze_performance()

if __name__ == "__main__":
    main()
