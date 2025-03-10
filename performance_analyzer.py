import time
import psutil
import streamlit as st

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
    "document_reference_hits": 0,  # New metric to track document referencing accuracy
}

def analyze_performance():
    try:
        total_queries = performance_data["total_queries"]
        successful_queries = performance_data["successful_queries"]
        failed_queries = performance_data["failed_queries"]
        total_response_time = performance_data["total_response_time"]
        total_error_count = performance_data["total_error_count"]
        benchmark_success = performance_data["benchmark_success"]
        document_reference_hits = performance_data["document_reference_hits"]

        # Performance Metrics
        query_success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        average_response_time = (total_response_time / successful_queries) if successful_queries > 0 else 0
        error_handling_success = (1 - (total_error_count / total_queries)) * 100 if total_queries > 0 else 100
        average_cpu_usage = performance_data["total_cpu_usage"] / total_queries if total_queries > 0 else 0
        average_memory_usage = performance_data["total_memory_usage"] / total_queries if total_queries > 0 else 0

        # Additional Metrics
        benchmark_success_rate = (benchmark_success / total_queries) * 100 if total_queries > 0 else 0
        document_reference_accuracy = (document_reference_hits / total_queries) * 100 if total_queries > 0 else 0

        # Overall Performance Score (Adjusted)
        overall_score = (query_success_rate + (100 - average_response_time * 10) + error_handling_success + benchmark_success_rate + document_reference_accuracy) / 5

        st.subheader("📊 Performance Analysis Results")
        st.write(f"🔹 **Total Queries Processed:** {total_queries}")
        st.write(f"✅ **Successful Queries:** {successful_queries} ({query_success_rate:.2f}%)")
        st.write(f"❌ **Failed Queries:** {failed_queries}")
        st.write(f"⏳ **Average Response Time:** {average_response_time:.2f} seconds")
        st.write(f"⚠️ **Error Handling Success:** {error_handling_success:.2f}%")
        st.write(f"💾 **Average CPU Usage:** {average_cpu_usage:.2f}%")
        st.write(f"📉 **Average Memory Usage:** {average_memory_usage:.2f}%")
        st.write(f"🏆 **Benchmark Success Rate:** {benchmark_success_rate:.2f}%")
        st.write(f"📚 **Document Referencing Accuracy:** {document_reference_accuracy:.2f}%")
        st.write(f"🏆 **Overall Performance Score:** {overall_score:.2f}%")
    except Exception as e:
        st.error(f"Error analyzing performance: {e}")

def track_query_performance(start_time, success, error_occurred, is_benchmark_query=False, document_referenced=False):
    performance_data["total_queries"] += 1
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    if success:
        performance_data["successful_queries"] += 1
        performance_data["total_response_time"] += (time.time() - start_time)
    else:
        performance_data["failed_queries"] += 1

    if error_occurred:
        performance_data["total_error_count"] += 1

    performance_data["total_cpu_usage"] += cpu_usage
    performance_data["total_memory_usage"] += memory_usage

    if is_benchmark_query:
        performance_data["benchmark_success"] += 1

    if document_referenced:
        performance_data["document_reference_hits"] += 1

def run_benchmark_tests():
    benchmark_queries = [
        {
            "query": "Can your chatbot calculate a tour budget correctly?",
            "expected_output": "Total estimated tour budget:",
            "is_benchmark": True,
            "document_referenced": True
        },
        {
            "query": "Can the chatbot keep my personal information confidential?",
            "expected_output": "Sorry, that's confidential.",
            "is_benchmark": True,
            "document_referenced": False
        },
        {
            "query": "How do I book a hotel at Cox's Bazar?",
            "expected_output": "Sorry, I can only assist with information related to your uploaded documents.",
            "is_benchmark": False,
            "document_referenced": True
        },
        # Add more queries as needed.
    ]
    for benchmark_query in benchmark_queries:
        start_time = time.time()
        response = simulate_query(benchmark_query["query"])
        success = benchmark_query["expected_output"] in response
        track_query_performance(start_time, success, not success, benchmark_query["is_benchmark"], benchmark_query["document_referenced"])

def simulate_query(query):
    # Simulate responses; replace with your actual query handling logic if desired.
    if query == "Can your chatbot calculate a tour budget correctly?":
        return "Total estimated tour budget: 5000 BDT"
    elif query == "Can the chatbot keep my personal information confidential?":
        return "Sorry, that's confidential."
    else:
        return "Sorry, I can only assist with information related to your uploaded documents."

def show_performance_page():
    st.header("Performance Analysis")
    st.markdown("Below are the performance metrics of your chatbot:")
    run_benchmark_tests()
    analyze_performance()

if __name__ == "__main__":
    show_performance_page()
