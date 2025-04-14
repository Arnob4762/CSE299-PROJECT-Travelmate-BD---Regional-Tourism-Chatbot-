import time
import psutil
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize performance data dictionary
performance_data = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "total_response_time": 0,
    "total_error_count": 0,
    "total_cpu_usage": 0,
    "total_memory_usage": 0,
    "benchmark_success": 0,  
    "document_reference_hits": 0,  
}

# List of benchmark queries and expected answers (for simplicity)
benchmark_queries = {
    "Tell me about the best hotels in Cox's Bazar?": "Hotel X, Hotel Y, Hotel Z",  # Example
}

def track_query_performance(success: bool, response_time: float, document_reference_hit: bool, is_benchmark: bool = False):
    performance_data["total_queries"] += 1
    performance_data["total_response_time"] += response_time
    performance_data["total_cpu_usage"] += psutil.cpu_percent(interval=0.1)
    performance_data["total_memory_usage"] += psutil.virtual_memory().used / (1024 * 1024)

    if success:
        performance_data["successful_queries"] += 1
        if document_reference_hit:
            performance_data["document_reference_hits"] += 1
        if is_benchmark:
            performance_data["benchmark_success"] += 1
    else:
        performance_data["failed_queries"] += 1
        performance_data["total_error_count"] += 1

def analyze_performance():
    try:
        total_queries = performance_data["total_queries"]
        successful_queries = performance_data["successful_queries"]
        failed_queries = performance_data["failed_queries"]
        total_response_time = performance_data["total_response_time"]
        total_error_count = performance_data["total_error_count"]
        benchmark_success = performance_data["benchmark_success"]
        document_reference_hits = performance_data["document_reference_hits"]

        query_success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        average_response_time = (total_response_time / successful_queries) if successful_queries > 0 else 0
        error_handling_success = (1 - (total_error_count / total_queries)) * 100 if total_queries > 0 else 100
        average_cpu_usage = performance_data["total_cpu_usage"] / total_queries if total_queries > 0 else 0
        average_memory_usage = performance_data["total_memory_usage"] / total_queries if total_queries > 0 else 0
        benchmark_rate = (benchmark_success / total_queries) * 100 if total_queries > 0 else 0

        return (
            f"Total Queries: {total_queries}\n"
            f"✅ Successful Queries: {successful_queries} ({query_success_rate:.2f}%)\n"
            f"❌ Failed Queries: {failed_queries}\n"
            f"⚡ Average Response Time: {average_response_time:.2f} seconds\n"
            f"🧠 Error Handling Success: {error_handling_success:.2f}%\n"
            f"🧮 Average CPU Usage: {average_cpu_usage:.2f}%\n"
            f"🗄️ Average Memory Usage: {average_memory_usage:.2f} MB\n"
            f"📄 Document Reference Hits: {document_reference_hits}\n"
            f"📌 Benchmark Success: {benchmark_success} ({benchmark_rate:.2f}%)"
        )

    except Exception as e:
        return f"Error analyzing performance: {e}"

def process_query(query):
    start_time = time.time()
    expected_answer = benchmark_queries.get(query)
    response = "Hotel X, Hotel Y, Hotel Z"  # Dummy response

    success = True
    if expected_answer:
        vector_response = np.array([ord(c) for c in response]).reshape(1, -1)
        vector_expected = np.array([ord(c) for c in expected_answer]).reshape(1, -1)
        similarity = cosine_similarity(vector_response, vector_expected)
        success = similarity >= 0.9

    response_time = time.time() - start_time

    track_query_performance(
        success=success,
        response_time=response_time,
        document_reference_hit=True,
        is_benchmark=bool(expected_answer)
    )

    return response

# Gradio interface for testing
performance_interface = gr.Interface(
    fn=analyze_performance,
    inputs=[],
    outputs="text",
    title="Performance Analyzer"
)

if __name__ == "__main__":
    performance_interface.launch()
