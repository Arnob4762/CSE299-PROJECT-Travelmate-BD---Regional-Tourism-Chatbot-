import time

# ✅ Import shared utilities from core_utils.py
from core_utils import get_file_text, app_state, update_performance_stats
from app import chat_with_documents


# Performance metrics tracking variables
total_queries = 0
accurate_responses = 0
total_response_time = 0

# Optional: wrapper if needed elsewhere
def track_query_performance(user_input, files):
    return analyze_performance(user_input, files)

# Analyze the performance of a query
def analyze_performance(user_input, files):
    start_time = time.time()

    response = chat_with_documents(user_input, files)

    response_time = time.time() - start_time
    is_accurate = False

    expected_answer = retrieve_expected_answer(user_input, files)
    if expected_answer and response.lower() in expected_answer.lower():
        is_accurate = True

    update_performance_stats(response_time, is_accurate)

    from core_utils import get_performance_report
    performance_report = get_performance_report()

    return performance_report


# Get full file content (naively used for checking accuracy)
def retrieve_expected_answer(user_input, files):
    text, _ = get_file_text(files)
    return text
