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
    "accurate_responses": 0,  # Accuracy placeholder (update your accuracy logic as needed)
    "total_response_time": 0,
    "faiss_index": None,
    "text_chunks": [],
    "meta_chunks": [],
    "hf_pipeline": None,
    "BASIC_RESPONSES": {}
}

# File handling
def get_file_text(files):
    text, metadata = "", []
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
    return text.strip(), metadata

# Processing
def process_and_store_chunks(text, metadata):
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    chunk_meta = [metadata[i % len(metadata)] for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    app_state["text_chunks"] = chunks
    app_state["meta_chunks"] = chunk_meta
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    app_state["faiss_index"] = index

# Retrieval
def retrieve_context(query, k=5):
    if not app_state["faiss_index"] or not app_state["text_chunks"]:
        return []
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = app_state["faiss_index"].search(query_embedding, k)
    return [(app_state["text_chunks"][i], app_state["meta_chunks"][i]) for i in I[0]]

# Performance Tracking
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
        f"- Total Queries: {total}\n"
        f"- Accurate Responses: {app_state['accurate_responses']}\n"
        f"- Accuracy: {accuracy:.2f}%\n"
        f"- Average Response Time: {avg_time:.2f} seconds"
    )

# Chatbot core
def chat_with_documents(user_input, files):
    start_time = time.time()
    # Use basic responses if applicable
    key = user_input.lower().strip()
    if key in app_state.get("BASIC_RESPONSES", {}):
        response = app_state["BASIC_RESPONSES"][key]
    else:
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
        hf_pipeline = app_state["hf_pipeline"]
        # Generate answer using the pipeline
        gen_result = hf_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]
        response = gen_result["generated_text"] if isinstance(gen_result, dict) else gen_result

    elapsed = time.time() - start_time
    # Here we update performance stats. Currently, we set is_accurate as False.
    update_performance_stats(elapsed, False)
    app_state["chat_history"].append((user_input, response))
    return f"**Response:**\n{response}"
