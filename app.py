import gradio as gr
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from tour_budget import show_budget_calculator
import performance_analyzer  # Avoid direct function import to prevent circular imports
from core_utils import (
    get_file_text, process_and_store_chunks,
    retrieve_context, BASIC_RESPONSES, app_state
)

# Load environment variables
load_dotenv()

# Load DeepSeek model pipeline once globally
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

# Main chatbot function — moved to core_utils.py in actual fix
def chat_with_documents(user_input, files):
    if user_input.lower().strip() in BASIC_RESPONSES:
        response = BASIC_RESPONSES[user_input.lower().strip()]
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
        response = hf_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]
        response = response["generated_text"] if isinstance(response, dict) else response

    app_state["chat_history"].append((user_input, response))
    return f"**Response:**\n{response}"

# Export this function so performance_analyzer can import it without circular issue
from core_utils import __dict__ as core_utils_namespace
core_utils_namespace["chat_with_documents"] = chat_with_documents

# Gradio Interfaces
def chatbot_tab():
    with gr.Column():
        chatbot_input = gr.Textbox(label="Ask a question about your documents:")
        chatbot_files = gr.File(label="Upload PDF or DOCX", file_types=['.pdf', '.docx'], file_count="multiple")
        chatbot_output = gr.Markdown()
        chatbot_button = gr.Button("Get Answer")
        chatbot_button.click(fn=chat_with_documents, inputs=[chatbot_input, chatbot_files], outputs=chatbot_output)
    return chatbot_input, chatbot_files, chatbot_output

def performance_tab():
    with gr.Column() as performance_interface:
        input_box = gr.Textbox(label="Enter a query to analyze:")
        file_input = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"], file_count="multiple")
        output_box = gr.Markdown()
        analyze_button = gr.Button("Analyze Performance")
        analyze_button.click(fn=performance_analyzer.analyze_performance, inputs=[input_box, file_input], outputs=output_box)
    return performance_interface

def guide_map_tab():
    return '<iframe src="https://arnob4762.github.io/tour-guide/" width="100%" height="600px" style="border:none;"></iframe>'

# Launch Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🧭 Regional Tourism Chatbot")

    with gr.Tab("📄 Chatbot"):
        chatbot_tab()

    with gr.Tab("💰 Budget Calculator"):
        show_budget_calculator()

    with gr.Tab("📊 Performance Analyzer"):
        performance_tab()

    with gr.Tab("🗺️ Guide Map"):
        gr.HTML(guide_map_tab())

if __name__ == "__main__":
    demo.launch(share=True)
