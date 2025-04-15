import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Performance metrics tracking variables
total_queries = 0
accurate_responses = 0
total_response_time = 0

# Function to track query performance
def track_query_performance(user_input, files):
    return analyze_performance(user_input, files)

# Function to analyze performance
def analyze_performance(user_input, files):
    global total_queries, accurate_responses, total_response_time

    start_time = time.time()

    # Process query and retrieve answer
    response = chat_with_documents(user_input, files)

    response_time = time.time() - start_time
    total_response_time += response_time
    total_queries += 1

    # Accuracy check: compare response with the expected content from the document
    expected_answer = retrieve_expected_answer(user_input, files)
    if expected_answer and response.lower() in expected_answer.lower():
        accurate_responses += 1

    # Calculate accuracy and average response time
    accuracy = (accurate_responses / total_queries) * 100 if total_queries > 0 else 0
    avg_response_time = total_response_time / total_queries if total_queries > 0 else 0

    performance_report = (
        f"Total Queries: {total_queries}\n"
        f"Accurate Responses: {accurate_responses}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Average Response Time: {avg_response_time:.2f} seconds"
    )

    return performance_report

# Function to retrieve expected answer from uploaded files
def retrieve_expected_answer(user_input, files):
    text, _ = get_file_text(files)
    return text
