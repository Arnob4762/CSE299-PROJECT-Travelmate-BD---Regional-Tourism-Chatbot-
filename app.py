import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Prevent NoneType errors
                text += page_text + "\n\n"  # Add spacing between pages
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Use a single separator
        chunk_size=500,
        chunk_overlap=100,  # Ensure overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("REGIONAL TOURISM CHATBOT :books:")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process") and pdf_docs:
            # Extract PDF text
            raw_text = get_pdf_text(pdf_docs)
            st.session_state["raw_text"] = raw_text  # Store extracted text in session state
            
            # Split into overlapping chunks
            text_chunks = get_text_chunks(raw_text)
            st.session_state["text_chunks"] = text_chunks  # Store chunks in session state
            
            st.success("Processing complete! Click 'View Chunks' to see the extracted text chunks.")

    if "text_chunks" in st.session_state:
        if st.button("View Chunks"):
            st.session_state["show_chunks"] = True

    if st.session_state.get("show_chunks", False):
        show_chunks_page()

def show_chunks_page():
    """Displays the chunked text on a new page."""
    st.title("Extracted Text Chunks")

    text_chunks = st.session_state.get("text_chunks", [])
    if not text_chunks:
        st.error("No chunks available. Please upload and process a PDF first.")
        return

    st.subheader(f"Total Chunks: {len(text_chunks)}")

    for i, chunk in enumerate(text_chunks):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)

    if st.button("Back to Main Page"):
        st.session_state["show_chunks"] = False
        st.rerun()  # Refresh the page to return to the main view

if __name__ == '__main__':
    main()
