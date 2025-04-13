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
    "benchmark_success": 0,  # To track benchmark query success
    "document_reference_hits": 0,  # New metric to track document referencing accuracy
}

# List of benchmark queries and expected answers (for simplicity)
benchmark_queries = {
    "Tell me about the best hotels in Cox's Bazar?": "Hotel X, Hotel Y, Hotel Z",  # Example
}

def track_query_performance(success: bool, response_time: float, document_reference_hit: bool, is_benchmark: bool = False):
    """Tracks performance metrics after each query execution."""
    performance_data["total_queries"] += 1
    performance_data["total_response_time"] += response_time
    performance_data["total_cpu_usage"] += psutil.cpu_percent(interval=0.1)
    performance_data["total_memory_usage"] += psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

    if success:
        performance_data["successful_queries"] += 1
        if document_reference_hit:
            performance_data["document_reference_hits"] += 1
        if is_benchmark:
            performance_data["benchmark_success"] += 1  # Increment benchmark success if it's a benchmark query
    else:
        performance_data["failed_queries"] += 1
        performance_data["total_error_count"] += 1

def analyze_performance():
    """Displays chatbot performance analytics on the main page."""
    try:
        total_queries = performance_data["total_queries"]
        successful_queries = performance_data["successful_queries"]
        failed_queries = performance_data["failed_queries"]
        total_response_time = performance_data["total_response_time"]
        total_error_count = performance_data["total_error_count"]
        benchmark_success = performance_data["benchmark_success"]
        document_reference_hits = performance_data["document_reference_hits"]

        # Calculate metrics
        query_success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        average_response_time = (total_response_time / successful_queries) if successful_queries > 0 else 0
        error_handling_success = (1 - (total_error_count / total_queries)) * 100 if total_queries > 0 else 100
        average_cpu_usage = performance_data["total_cpu_usage"] / total_queries if total_queries > 0 else 0
        average_memory_usage = performance_data["total_memory_usage"] / total_queries if total_queries > 0 else 0

        st.header("Performance Analytics")
        st.write(f"Total Queries: {total_queries}")
        st.write(f"Successful Queries: {successful_queries} ({query_success_rate:.2f}%)")
        st.write(f"Failed Queries: {failed_queries}")
        st.write(f"Average Response Time: {average_response_time:.2f} seconds")
        st.write(f"Error Handling Success: {error_handling_success:.2f}%")
        st.write(f"Average CPU Usage: {average_cpu_usage:.2f}%")
        st.write(f"Average Memory Usage: {average_memory_usage:.2f} MB")
        st.write(f"Document Reference Hits: {document_reference_hits}")
        benchmark_rate = (benchmark_success / total_queries) * 100 if total_queries > 0 else 0
        st.write(f"Benchmark Success: {benchmark_success} ({benchmark_rate:.2f}%)")
    except Exception as e:
        st.error(f"Error analyzing performance: {e}")

# Now track response time for each query and benchmark success
def process_query(query):
    """Function that processes the query and measures the time taken."""
    start_time = time.time()  # Start timer

    # Check if the query is a benchmark query and get the expected answer
    expected_answer = benchmark_queries.get(query)

    # Simulate query processing (replace this with actual query processing)
    response = "Query result based on document"  # This should be your chatbot's response logic

    # Compare response with the expected answer for benchmark queries
    success = False
    if expected_answer:
        success = (response == expected_answer)

    # Calculate response time
    response_time = time.time() - start_time  # Elapsed time

    # Track performance, including benchmark success
    track_query_performance(success=success, response_time=response_time, document_reference_hit=False, is_benchmark=bool(expected_answer))

    return response

# Example usage
if __name__ == "__main__":
    query = "Tell me about the best hotels in Cox's Bazar?"
    response = process_query(query)
    st.write(response)
