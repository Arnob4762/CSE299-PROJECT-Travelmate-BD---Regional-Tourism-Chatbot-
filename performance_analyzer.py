# performance_analyzer.py

import time
import streamlit as st

# Initialize performance data dictionary
performance_data = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "total_response_time": 0,
    "total_error_count": 0,
}

# Function to perform the performance analysis
def analyze_performance():
    try:
        query_performance = (performance_data["successful_queries"] / performance_data["total_queries"]) * 100 if performance_data["total_queries"] > 0 else 0
        average_response_time = performance_data["total_response_time"] / performance_data["total_queries"] if performance_data["total_queries"] > 0 else 0
        error_handling = (1 - (performance_data["total_error_count"] / performance_data["total_queries"])) * 100 if performance_data["total_queries"] > 0 else 0

        # Calculate overall score (you can adjust this formula as needed)
        overall_score = (query_performance + (100 - average_response_time) + error_handling) / 3

        # Display the results
        st.subheader("Performance Analysis Results")
        st.write(f"Total Queries Processed: {performance_data['total_queries']}")
        st.write(f"Successful Queries: {performance_data['successful_queries']} ({query_performance:.2f}%)")
        st.write(f"Failed Queries: {performance_data['failed_queries']}")
        st.write(f"Average Response Time: {average_response_time:.2f} seconds")
        st.write(f"Error Handling Success: {error_handling:.2f}%")
        st.write(f"Overall Performance Score: {overall_score:.2f}%")
        
    except Exception as e:
        st.error(f"Error analyzing performance: {e}")


# Function to track query performance and response time
def track_query_performance(start_time, success, error_occurred):
    performance_data["total_queries"] += 1
    if success:
        performance_data["successful_queries"] += 1
    else:
        performance_data["failed_queries"] += 1
    
    # Update response time
    if success:
        performance_data["total_response_time"] += (time.time() - start_time)
    
    # Update error count
    if error_occurred:
        performance_data["total_error_count"] += 1
