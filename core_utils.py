import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Shared state
app_state = {
    "chat_history": [],
    "total_queries": 0,
    "accurate_responses": 0,
    "total_response_time": 0,
    "faiss_index": None,
    "text_chunks": [],
    "meta_chunks": [],
    "hf_pipeline": None,
    "BASIC_RESPONSES": {}
}

# ---------------------------
# File handling
# ---------------------------
def get_file_text(files):
    text, metadata = "", []
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                try:
                    content = page.extract_text() or ""
                    text += content + "\n\n"
                    metadata.append((file.name, f"page {i + 1}"))
                except Exception as e:
                    print(f"Warning: Failed to read page {i + 1} of {file.name} – {e}")
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for i, para in enumerate(doc.paragraphs):
                text += para.text + "\n\n"
                metadata.append((file.name, f"paragraph {i + 1}"))
    return text.strip(), metadata

# ---------------------------
# Chunk processing
# ---------------------------
def process_and_store_chunks(text, metadata):
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Align metadata
    if len(metadata) < len(chunks):
        metadata += [metadata[-1]] * (len(chunks) - len(metadata))
    chunk_meta = metadata[:len(chunks)]

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    app_state["text_chunks"] = chunks
    app_state["meta_chunks"] = chunk_meta
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    app_state["faiss_index"] = index

# ---------------------------
# Context retrieval
# ---------------------------
def retrieve_context(query, k=5):
    if not app_state["faiss_index"] or not app_state["text_chunks"]:
        return []
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = app_state["faiss_index"].search(query_embedding, k)

    results = []
    for i in I[0]:
        if i >= 0 and i < len(app_state["text_chunks"]):
            chunk = app_state["text_chunks"][i]
            meta = app_state["meta_chunks"][i]
            if chunk.strip():
                results.append((chunk, meta))
    return results

# ---------------------------
# Performance tracking
# ---------------------------
def update_performance_stats(response_time, is_accurate):
    app_state["total_queries"] += 1
    app_state["total_response_time"] += response_time
    if is_accurate:
        app_state["accurate_responses"] += 1

def get_performance_report():
    total = app_state["total_queries"]
    accuracy = (app_state["accurate_responses"] / total) * 100 if total else 0
    avg_time = app_state["total_response_time"] / total if total else 0
    return (
        f"**Performance Summary:**\n\n"
        f"Total Queries: {total}\n"
        f"Accurate Responses: {app_state['accurate_responses']}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Average Response Time: {avg_time:.2f} seconds"
    )

# ---------------------------
# Chat logic
# ---------------------------
def chat_with_documents(user_input, files):
    start_time = time.time()
    key = user_input.lower().strip()

    if key in app_state.get("BASIC_RESPONSES", {}):
        response = app_state["BASIC_RESPONSES"][key]
    else:
        # Load or reset documents
        if files:
            text, meta = get_file_text(files)
            process_and_store_chunks(text, meta)
        elif not app_state["faiss_index"]:
            return "**Response:** Please upload a document first."

        # Retrieve context
        results = retrieve_context(user_input)
        if results:
            top_chunks = results[:2]  # Use top 2 for more richness
            context = "\n\n".join(chunk for chunk, _ in top_chunks)
            references = ", ".join([f"{filename}, {page}" for _, (filename, page) in top_chunks])
            reference_tag = f"[{references}]"
        else:
            context = ""
            reference_tag = ""

        # Build prompt
        prompt = (
            f"Answer the following question using ONLY the provided context below.\n"
            f"Be direct and concise. Do NOT use outside knowledge.\n"
            f"End your answer with the source reference in brackets.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}\n"
            f"Answer:"
        )

        hf_pipeline = app_state["hf_pipeline"]
        gen_result = hf_pipeline(prompt, max_new_tokens=150, do_sample=False, temperature=0.3)[0]
        full_output = gen_result["generated_text"] if isinstance(gen_result, dict) else gen_result

        if "Answer:" in full_output:
            response = full_output.split("Answer:")[-1].strip()
        else:
            response = full_output.strip()

        if reference_tag and reference_tag not in response:
            response += f" {reference_tag}"

    # Log stats
    elapsed = time.time() - start_time
    update_performance_stats(elapsed, False)
    app_state["chat_history"].append((user_input, response))

    return f"**Response:** {response}"

# ---------------------------
# Manual Feedback
# ---------------------------
def on_feedback_accurate():
    app_state["accurate_responses"] += 1
    return "Feedback recorded: Accurate"

def on_feedback_inaccurate():
    return "Feedback recorded: Inaccurate"

# ---------------------------
# Auto Feedback (Optional)
# ---------------------------
def is_response_accurate(user_input, chatbot_response):
    expected_keywords = {
        "cox's bazar": ["beach", "sea", "coastal", "inani", "laboni", "himchari"],
        "sundarbans": ["mangrove", "tiger", "forest", "boat"],
    }
    for keyword, required_words in expected_keywords.items():
        if keyword in user_input.lower():
            return any(word in chatbot_response.lower() for word in required_words)
    return False

def on_bot_response(user_input, chatbot_response):
    app_state["total_queries"] += 1
    if is_response_accurate(user_input, chatbot_response):
        app_state["accurate_responses"] += 1
    return get_performance_report()
