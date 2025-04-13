import streamlit as st
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from tour_budget import show_budget_calculator  # Budget Estimator
from performance_analyzer import track_query_performance, analyze_performance  # Performance tracker

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
                metadatas=[meta]
            )
    except Exception as e:
        st.error(f"Error storing chunks in ChromaDB: {e}")

def query_chromadb(query):
    try:
        start_time = time.time()  # Record start time
        if collection.count() == 0:
            return []
        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        
        # Debug: display metadata retrieved from query results
        st.write("References:", metas)
        
        # Track query performance
        query_duration = time.time() - start_time
        track_query_performance(
            success=True,
            response_time=query_duration,
            document_reference_hit=bool(docs)
        )
        return [(doc, meta) for doc, meta in zip(docs, metas)]
    except Exception as e:
        query_duration = time.time() - start_time
        track_query_performance(
            success=False,
            response_time=query_duration,
            document_reference_hit=False
        )
        st.error(f"Error querying ChromaDB: {e}")
        return []

# ----- GUIDE MAP -----
def show_guide_map():
    st.header("Tour Guide Map")
    st.markdown(
        '<iframe src="https://arnob4762.github.io/tour-guide/" width="100%" height="600px" style="border:none;"></iframe>',
        unsafe_allow_html=True
    )

# ----- CHATBOT PAGE -----
def chatbot_page():
    st.header("  ")
    user_input = st.text_input("Ask a question about your documents:")

    if user_input:
        lower_input = user_input.lower().strip()
        if lower_input in BASIC_RESPONSES:
            st.write(BASIC_RESPONSES[lower_input])
        else:
            relevant_chunks = query_chromadb(user_input)
            context = ""
            references_list = []
            for doc, meta in relevant_chunks:
                if isinstance(meta, dict):
                    doc_name = meta.get("document_name", "Unknown Document")
                    page = meta.get("page_number")
                    paragraph = meta.get("paragraph_number")
                    ref_str = f"{doc_name}"
                    if page:
                        ref_str += f", page {page}"
                    if paragraph:
                        ref_str += f", paragraph {paragraph}"
                    references_list.append(ref_str)
                    context += f"[{ref_str}]: {str(doc)}\n\n"
                else:
                    context += str(doc) + "\n\n"

            # # Debug: show references list for checking
            # st.write("References List:", references_list)

            # if not context:
            #     st.warning("No relevant data found.")
            #     context = "No relevant data found."
            
            # Modified prompt with instructions for chain-of-thought and final answer
            prompt = (
                f"Context:\n{context}\n\n"
                f"User Question: {user_input}\n\n"
                "Please provide your internal chain-of-thought (prefixed with 'Thinking:') and then your final answer (prefixed with 'Final Answer:')."
            )
            
            response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])["message"]["content"]
            
            # Split response by the delimiter "Final Answer:"
            if "Final Answer:" in response:
                parts = response.split("Final Answer:")
                chain_of_thought = parts[0].replace("Thinking:", "").strip()
                final_answer = parts[1].strip()
            else:
                chain_of_thought = ""
                final_answer = response.strip()

            # --- Modified Section Start ---
            # Display the internal chain-of-thought in light ash color, 10pt Times New Roman
            if chain_of_thought:
                st.markdown(
                    f"<div style='color: #D3D3D3; font-size: 10pt; font-family: \"Times New Roman\", Times, serif; margin-bottom: 10px;'>"
                    f"<em>&lt;thinking.....&gt;<br>{chain_of_thought}</em></div>",
                    unsafe_allow_html=True
                )
            
            # Format the final answer as a bullet list in bright white, bold, 14pt Times New Roman
            final_answer_lines = [line.strip() for line in final_answer.split('\n') if line.strip()]
            formatted_final_answer = "<ul style='list-style-type: disc; margin-left: 20px;'>"
            for line in final_answer_lines:
                formatted_final_answer += f"<li>{line}</li>"
            formatted_final_answer += "</ul>"
            st.markdown(
                f"<div style='color: #FFFFFF; font-size: 14pt; font-family: \"Times New Roman\", Times, serif; font-weight: bold; margin-bottom: 20px;'>"
                f"{formatted_final_answer}</div>",
                unsafe_allow_html=True
            )
            
            # Display the document references as "References List:" below the final answer


            # st.markdown("<hr>", unsafe_allow_html=True)
            # st.markdown(
            #     "<div style='font-size: 0.9em; margin-bottom: 10px;'><strong>References List:</strong></div>",
            #     unsafe_allow_html=True
            # )
            # if references_list:
            #     formatted_refs = "<ul style='list-style-type: disc; margin-left: 20px;'>"
            #     for ref in set(references_list):
            #         formatted_refs += f"<li>{ref}</li>"
            #     formatted_refs += "</ul>"
            #     st.markdown(formatted_refs, unsafe_allow_html=True)
            # else:
            #     st.markdown("<div style='font-size: 0.9em;'>No document references found.</div>", unsafe_allow_html=True)
            
            # Add a gap before the document upload box
            st.markdown("<br><br>", unsafe_allow_html=True)
            # --- Modified Section End ---

    # Document Upload Section
    uploaded_files = st.file_uploader("Upload your documents (PDF/Word)", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        text, metadata = get_file_text(uploaded_files)
        st.write("Extracted Metadata:", metadata)  # Debug: Verify metadata extraction
        chunks = get_text_chunks(text, metadata)
        store_chunks_in_chromadb(chunks)
        st.success("Documents uploaded and processed successfully!")

# ----- BUDGET CALCULATOR PAGE -----
def budget_calculator_page():
    st.header("Tour Budget Calculator")
    show_budget_calculator()  # This function now displays the calculator in the main UI

# ----- MAIN APP LAYOUT -----
def main():
    st.title("Regional Tourism Chatbot")

    # Sidebar with navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Chatbot", "Budget Calculator", "Performance Analyzer", "Guide Map"])
    
    if page == "Chatbot":
        chatbot_page()
    elif page == "Guide Map":
        show_guide_map()
    elif page == "Budget Calculator":
        budget_calculator_page()
    elif page == "Performance Analyzer":
        analyze_performance()

if __name__ == "__main__":
    main()
