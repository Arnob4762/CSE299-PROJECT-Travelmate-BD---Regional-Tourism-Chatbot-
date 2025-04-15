import time
from core_utils import get_file_text, app_state, update_performance_stats, get_performance_report, chat_with_documents

# Analyze the performance of a query
def analyze_performance(user_input, files):
    start_time = time.time()
    response = chat_with_documents(user_input, files)
    response_time = time.time() - start_time

    expected_answer = retrieve_expected_answer(user_input, files)
    is_accurate = expected_answer and response.lower() in expected_answer.lower()
    update_performance_stats(response_time, is_accurate)

    return get_performance_report()

# Naive expected response
def retrieve_expected_answer(user_input, files):
    text, _ = get_file_text(files)
    return text
