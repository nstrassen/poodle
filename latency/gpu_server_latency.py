import json
import os.path
import time
from statistics import median

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from data.imdb.reduced_imdb import load_imdb_data
from bert_accuracy.llama_7b import load_llama_model


def measure_llm_latency(data_path, batch_size=1, warm_up_cycles=3, measurement_cycles=10, results_path=None,
                        wrapper_prompt=""):
    model_name = "NousResearch/Llama-2-7b-chat-hf"  # replace with a public accessible variant
    tokenizer, model = load_llama_model(model_name)

    return measure_model_latency(model, tokenizer, data_path, "llm", batch_size, warm_up_cycles,
                                 measurement_cycles, results_path, wrapper_prompt)


def measure_bert_latency(data_path, batch_size=1, warm_up_cycles=3, measurement_cycles=10, results_path=None):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    return measure_model_latency(model, tokenizer, data_path, "bert", batch_size, warm_up_cycles,
                                 measurement_cycles, results_path)


def measure_model_latency(model, tokenizer, data_path, model_type, batch_size=1, warm_up_cycles=3,
                          measurement_cycles=10, results_path=None, wrapper_prompt=""):
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts, labels = load_imdb_data(data_path)

    # Use the same text for all batch items
    texts = [texts[0]] * 2048

    def _measure_inference_latency(infer_fn):
        # Warm-up
        for _ in range(warm_up_cycles):
            with torch.no_grad():
                infer_fn()

        # Measure
        times = []
        for _ in range(measurement_cycles):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                infer_fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        return times

    if model_type == "bert":
        inputs = tokenizer(texts[:batch_size], truncation=True, padding=True, max_length=256, return_tensors='pt').to(
            device)

        times = _measure_inference_latency(lambda: model(**inputs))

        # Decode one sample output to confirm pipeline works
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(f"Sample prediction for batch size {batch_size}: {predictions[0].item()}")

    elif model_type == "llm":
        # Create batch of prompts
        # for BERT we use the first 256 tokens, for a fair comparison we also just want to give the first 256 even if the
        # LLama tokenizer will extend this according to what the model expects later
        # assume 1 word -> 1.3 tokens; 256 / 1.3 -> 197 words + ~20 words for prompt ->220
        prompts = [
            f"{wrapper_prompt} Classify the sentiment of the following movie review as positive or negative:\n\n{' '.join(text.split()[:220])}\n\nSentiment:"
            for text in texts[:batch_size]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            device)

        max_new_tokens = 50

        times = _measure_inference_latency(lambda: model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7))

        # Decode one sample to confirm output
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7)
        sample_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Sample output for batch size {batch_size}:\n{sample_output}...")

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    avg_latency = sum(times) / len(times)
    median_latency = median(times)
    results.append({
        "batch_size": batch_size,
        "avg_latency_sec": avg_latency,
        "median_latency_sec": median_latency,
        "latencies": times
    })

    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        filename = os.path.join(results_path, f"latency_results_batchsize_{batch_size}.json")
        with open(filename, "w") as f:
            json.dump(results, f)

    return results


if __name__ == '__main__':
    data_path = "/mount-fs/poodle/labeled-data/imdb/test-500-splits/test-500-1"
    warm_up_cycles = 3
    measurement_cycles = 10


    # Note: currently we only get consistent result if we
    # 1) use the same text for all batch items

    batch_sizes = [1, 2, 4, 8, 16]  # 32 out of memory
    llm_results_path = "/mount-fs/poodle/latency-experiments/llm"
    for b in batch_sizes:
        result = measure_llm_latency(data_path, b, warm_up_cycles, measurement_cycles, llm_results_path)
        print(result)
        torch.cuda.empty_cache()
        time.sleep(2)

    wrapper_prompt = """
    1. Identify the type of input the user is providing. 2. Infer what task you are expected to perform. 3. Solve the task that the user is giving you. USER REQUEST:
    """

    batch_sizes = [1]  # crashes for 2; probably wrapper prompt too long
    llm_results_path = "/mount-fs/poodle/latency-experiments/llm-wrapped"
    for b in batch_sizes:
        result = measure_llm_latency(data_path, b, warm_up_cycles, measurement_cycles, llm_results_path, wrapper_prompt)
        print(result)
        torch.cuda.empty_cache()
        time.sleep(2)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # 256 out of memory
    bert_results_path = "/mount-fs/poodle/latency-experiments/bert"
    for b in batch_sizes:
        result = measure_bert_latency(data_path, b, warm_up_cycles, measurement_cycles, bert_results_path)
        print(result)
        torch.cuda.empty_cache()
        time.sleep(2)
