import time

# ✅ Import shared utilities from core_utils.py
from core_utils import get_file_text, chatbot_history, chat_with_documents

# Performance metrics tracking variables
total_queries = 0
accurate_responses = 0
total_response_time = 0

# Optional: wrapper if needed elsewhere
def track_query_performance(user_input, files):
    return analyze_performance(user_input, files)

# Analyze the performance of a query
def analyze_performance(user_input, files):
    global total_queries, accurate_responses, total_response_time

    start_time = time.time()

    # Get chatbot response (this also updates chatbot_history)
    response = chat_with_documents(user_input, files)

    response_time = time.time() - start_time
    total_response_time += response_time
    total_queries += 1

    # Accuracy check: naive method — does response contain parts of the file text?
    expected_answer = retrieve_expected_answer(user_input, files)
    if expected_answer and response.lower() in expected_answer.lower():
        accurate_responses += 1

    # Stats
    accuracy = (accurate_responses / total_queries) * 100 if total_queries > 0 else 0
    avg_response_time = total_response_time / total_queries if total_queries > 0 else 0

    performance_report = (
        f"**Performance Summary:**\n\n"
        f"- Total Queries: {total_queries}\n"
        f"- Accurate Responses: {accurate_responses}\n"
        f"- Accuracy: {accuracy:.2f}%\n"
        f"- Average Response Time: {avg_response_time:.2f} seconds"
    )

    return performance_report

# Get full file content (naively used for checking accuracy)
def retrieve_expected_answer(user_input, files):
    text, _ = get_file_text(files)
    return text
