def time_for_llm_requests(num_requests):
    return num_requests / 12.7

def time_for_jitr(num_requests):
    llm = 5000 / 0.7
    bert = (num_requests - 5000) / 255.8
    return llm + bert

if __name__ == '__main__':
    for i in [10, 20, 30, 40, 50 , 60, 70, 80, 90, 100, 150, 200, 1000, 2000]:
        num_requests = i * 1000
        llm_time = time_for_llm_requests(num_requests)
        jitr_time = time_for_jitr(num_requests)
        print(f"Num requests: {num_requests}: LLM time: {llm_time:.2f}s, JITR time: {jitr_time:.2f}s, LLM throughput: {num_requests/llm_time:.2f} rps, JITR throughput: {num_requests/jitr_time:.2f} rps")
        if jitr_time < llm_time:
            print(f"Break-even point at {num_requests} requests")