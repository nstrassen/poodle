import json
import os


def get_latency_summary(results_dir, batch_sizes):

    latency_summary = {}
    for batch_size in batch_sizes:
        filepath = os.path.join(results_dir, f"latency_results_batchsize_{batch_size}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                batch_size = entry.get("batch_size")
                median_latency = entry.get("median_latency_sec")
                latency_summary[batch_size] = median_latency

    return latency_summary


def items_per_second(latencies):
    throughput = {}
    for batch_size, latency in latencies.items():
        if latency and latency > 0:
            throughput[batch_size] = (batch_size / latency)
        else:
            throughput[batch_size] = 0.0
    return throughput


if __name__ == '__main__':
    base_path = "results/latency-experiments/"
    bert_latencies = get_latency_summary(os.path.join(base_path, "bert"), [1, 2, 4, 8, 16, 32, 64, 128])
    llm_latencies = get_latency_summary(os.path.join(base_path, "llm"), [1, 2, 4, 8, 16])
    print("bert latencies", bert_latencies)
    print("llm latencies", llm_latencies)
    print("bert items per second", items_per_second(bert_latencies))
    print("llm items per second", items_per_second(llm_latencies))