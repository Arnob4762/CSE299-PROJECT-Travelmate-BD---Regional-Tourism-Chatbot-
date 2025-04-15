import gradio as gr
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from tour_budget import show_budget_calculator
import performance_analyzer
from core_utils import (
    chat_with_documents, app_state
)

# Load environment variables
load_dotenv()

# Load DeepSeek model pipeline once globally and inject into app_state
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
app_state["hf_pipeline"] = hf_pipeline

# Optional: Add static BASIC_RESPONSES
app_state["BASIC_RESPONSES"] = {
    "hi": "Hello! How can I assist you with your travel documents?",
    "hello": "Hi there! Ask me anything about your travel plans.",
    "bye": "Goodbye! Safe travels!"
}

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
        output_box = gr.Markdown(label="Performance Summary")
        refresh_button = gr.Button("Refresh Performance Summary")
        # Simply return the current performance report
        refresh_button.click(fn=performance_analyzer.get_performance_report, inputs=[], outputs=output_box)
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
