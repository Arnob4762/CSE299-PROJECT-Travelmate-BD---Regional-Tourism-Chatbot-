import streamlit as st  # Streamlit for web UI
from dotenv import load_dotenv  # Load environment variables
from PyPDF2 import PdfReader  # Read PDFs
import docx  # Read DOCX files
from langchain.text_splitter import CharacterTextSplitter  # Split text into chunks
import chromadb  # ChromaDB for vector storage
from sentence_transformers import SentenceTransformer  # For generating embeddings
import numpy as np  # To handle embeddings

# Load the embedding model (using a small, efficient model)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Use persistent storage
collection = chroma_client.get_or_create_collection(name="tourism_chatbot")  # Create or retrieve collection

# Function to extract text from uploaded PDF and DOCX files
def get_file_text(files):
    text = ""  # Store extracted text
    for file in files:
        if file.type == "application/pdf":  # If it's a PDF file
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Append text from each page
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # If it's a DOCX file
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n\n"  # Append text from each paragraph
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,  # Maximum 500 words per chunk
        chunk_overlap=100,  # Overlap ensures better context
        length_function=len
    )
    return text_splitter.split_text(text)  # Return list of chunks

# Function to generate embeddings using Sentence Transformers
def generate_embeddings(chunks):
    return embedding_model.encode(chunks, convert_to_numpy=True).tolist()  # Convert to list format

# Function to store text chunks and embeddings in ChromaDB
def store_chunks_in_chromadb(chunks):
    embeddings = generate_embeddings(chunks)  # Generate embeddings for each chunk

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],  # Unique ID for each chunk
            documents=[chunk],  # Store actual text
            embeddings=[embedding]  # Store embeddings
        )

# Function to check ChromaDB status from Sidebar
def check_chromadb_status():
    total_docs = collection.count()
    st.write(f"Total Chunks Stored in ChromaDB: {total_docs}")

# Function to retrieve and display stored embeddings
def view_embeddings():
    stored_data = collection.get(include=["documents", "embeddings"])  # Fetch both text and embeddings

    docs = stored_data.get("documents", [])
    embeddings = stored_data.get("embeddings", [])

    if len(docs) == 0 or len(embeddings) == 0:
        st.error("No embeddings found. Please process documents first.")
        return

    st.subheader("Stored Embeddings in ChromaDB")

    # Display text chunk and corresponding embedding
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        with st.expander(f"Chunk {i+1}"):
            st.write("Text Chunk:", doc)
            st.write("Embedding:", emb)  # Show the actual embedding vector

# Function to display extracted text chunks
def show_chunks_page():
    st.title("Extracted Text Chunks")

    text_chunks = st.session_state.get("text_chunks", [])
    if not text_chunks:
        st.error("No chunks available. Please upload and process a document first.")
        return

    st.subheader(f"Total Chunks: {len(text_chunks)}")

    # Display each chunk in an expandable section
    for i, chunk in enumerate(text_chunks):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)

    if st.button("Back to Main Page"):
        st.session_state["show_chunks"] = False
        st.rerun()

# Main function to run the Streamlit app
def main():
    load_dotenv()  # Load environment variables

    st.set_page_config(page_title="Chat with multiple PDFs and DOCs", page_icon=":world_map:")

    st.header("REGIONAL TOURISM CHATBOT")

    # Link to tourism guide map
    st.markdown("[Click here to view the Regional Tourism Guide Map](https://arnob4762.github.io/tour-guide/)")

    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload PDFs & DOCs", accept_multiple_files=True)

        if st.button("Process") and uploaded_files:
            raw_text = get_file_text(uploaded_files)
            st.session_state["raw_text"] = raw_text

            text_chunks = get_text_chunks(raw_text)
            st.session_state["text_chunks"] = text_chunks

            store_chunks_in_chromadb(text_chunks)  # Store embeddings in ChromaDB

            st.success("Processing complete! Click 'View Chunks' to see extracted text.")

        # Sidebar Buttons
        if st.button("Check ChromaDB"):
            check_chromadb_status()

        if st.button("View Stored Data"):
            stored_data = collection.get(include=["documents"])
            st.write("Stored Chunks in ChromaDB:", stored_data["documents"])

        if st.button("View Embeddings"):
            view_embeddings()

    if "text_chunks" in st.session_state:
        if st.button("View Chunks"):
            st.session_state["show_chunks"] = True

    if st.session_state.get("show_chunks", False):
        show_chunks_page()

# Run the app
if __name__ == '__main__':
    main()
