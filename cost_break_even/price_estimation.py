import matplotlib.pyplot as plt
import numpy as np

OUTPUT = "output"
INPUT = "input"
LLAMA_405B_TURBO = "Llama-405B-turbo"
LLAMA_70B_TURBO = "Llama-70B-turbo"
LLAMA_8B = "Llama-8B"
BERT_80M = "bert-80M"
GPT_4_1 = "gpt-4.1"
GPT_4_1_MINI = "gpt-4.1_mini"
GPT_4_1_NANO = "gpt-4.1_nano"

INPUT_TOKENS = "input_tokens"
OUTPUT_TOKENS = "output_tokens"
NUM_REQUESTS = "num_requests"
MODEL_ID = "model_id"

MODEL_PRICING_PER_1M = {  # prices in $/1M tokens
    # https://api.together.ai/models/togethercomputer/m2-bert-80M-32k-retrieval
    BERT_80M: {INPUT: 0.01, OUTPUT: 0.01},
    # Ref
    LLAMA_8B: {INPUT: 0.20, OUTPUT: 0.02},
    # https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo
    LLAMA_70B_TURBO: {INPUT: 0.88, OUTPUT: 0.88},
    # https://api.together.ai/models/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
    LLAMA_405B_TURBO: {INPUT: 3.50, OUTPUT: 3.50},
    # https://platform.openai.com/docs/pricing
    # - gpt 4.1: 2.00\$ (input), 8.00\$ (output), batch prices half
    GPT_4_1: {INPUT: 2, OUTPUT: 8},
    # - gpt 4.1-mini: 0.40\$ (input), 1.60\$ (output), batch prices half
    GPT_4_1_MINI: {INPUT: 0.4, OUTPUT: 1.6},
    # - gpt 4.1-nano: 0.10\$ (input), 0.40\$ (output), batch prices half
    GPT_4_1_NANO: {INPUT: 0.1, OUTPUT: 0.4},
}


def base_price_estimation(input_tokens, output_tokens, num_requests, model_id):
    # scale price to 1M tokens
    inp_token_price = MODEL_PRICING_PER_1M[model_id][INPUT] / 10 ** 6
    out_token_price = MODEL_PRICING_PER_1M[model_id][OUTPUT] / 10 ** 6

    price_per_request = inp_token_price * input_tokens + out_token_price * output_tokens

    price = price_per_request * num_requests

    return price


def config_price_estimation(config):
    return base_price_estimation(
        config[INPUT_TOKENS],
        config[OUTPUT_TOKENS],
        config[NUM_REQUESTS],
        config[MODEL_ID]
    )


def combined_price_estimation(large_config, small_config, total_requests, large_requests):
    large_config[NUM_REQUESTS] = large_requests
    small_config[NUM_REQUESTS] = total_requests - large_requests
    large_model_price = config_price_estimation(large_config)
    small_model_price = config_price_estimation(small_config)

    # TODO get better numbers here, probably cost will go lower
    # for now assume GPU time of 3 hours for model search and fine-tuning, have to validate
    # cost per hour ~1.2$/h -> ~4$
    compute_cost = 4

    return large_model_price + small_model_price + compute_cost


def collect_comparison_data(request_range_range, large_model, small_model):
    base_prices = {}
    optimized_prices = {}

    input_tokens = 403
    output_tokens = 8
    wrapped_input_tokens = 590
    wrapped_output_tokens = 40
    switch_after_n_items = 5000

    large_config = {INPUT_TOKENS: input_tokens, OUTPUT_TOKENS: output_tokens, MODEL_ID: large_model}
    small_config = {INPUT_TOKENS: wrapped_input_tokens, OUTPUT_TOKENS: wrapped_output_tokens, MODEL_ID: small_model}

    for num_requests in request_range_range:
        base_prices[num_requests] = \
            base_price_estimation(input_tokens, output_tokens, num_requests, large_model)
        optimized_prices[num_requests] = \
            combined_price_estimation(large_config, small_config, num_requests, switch_after_n_items)

    return base_prices, optimized_prices


def plot_price_savings(request_range_range, large_model, small_model):

    base_prices, optimized_prices = collect_comparison_data(request_range_range, large_model, small_model)

    requests = list(request_range_range)
    savings = [
        base_prices[n] - optimized_prices[n]
        for n in requests
    ]

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(4, 3))
    x = [n // 1000 for n in requests]
    # increase marker size for better visibility
    plt.plot(x, savings, marker='o', color='#43a2ca', linewidth=3, markersize=8)
    plt.xlabel("Requests (10^3)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=3)

    # ensure x axis starts at 0 and always shows exactly 3 ticks (starting with 0)
    x_max = max(x) if x else 0
    if x_max < 2:
        x_max = 2
    # add a tiny bit of space on the right so the last marker isn't cut off
    padding = max(1, x_max * 0.1)
    plt.xlim(left=0, right=x_max + padding)
    plt.xticks(np.linspace(0, x_max, 3))

    plt.ylabel("Savings ($)")

    y_min = min(savings) if savings else 0
    y_max = max(savings) if savings else 0
    if y_min == y_max:
        y_max = y_min + 1.0

    # Ensure 0 is always between y_min and y_max
    if y_max < 0:
        y_max = 0
    if y_min > 0:
        y_min = 0

    y_padding = max(0.5, (y_max - y_min) * 0.05)
    plt.ylim(bottom=y_min - y_padding, top=y_max + y_padding)
    plt.yticks([int(y_min - y_padding), 0, int(y_max + y_padding)])


    # plt.title(f"Price Savings: {large_model} vs Optimized ({small_model})")
    plt.tight_layout()
    plt.savefig(f"price_savings_{large_model}_vs_{small_model}.pdf")
    plt.show()


if __name__ == '__main__':
    input_tokens = 403
    output_tokens = 8
    wrapped_input_tokens = 590
    wrapped_output_tokens = 40
    num_requests = 1000000
    switch_after_n_items = 5000
    # for model_id in [BERT_80M, LLAMA_8B, LLAMA_70B_TURBO, LLAMA_405B_TURBO, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO]:
    #     for model_id in [BERT_80M, LLAMA_8B, LLAMA_70B_TURBO, LLAMA_405B_TURBO, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO]:
    #         price = base_price_estimation(input_tokens, output_tokens, num_requests, model_id)
    #     print(f"{model_id}: {price:.2f}$")

    for large_model, small_model in [(GPT_4_1_NANO, BERT_80M), (GPT_4_1_MINI, BERT_80M), (GPT_4_1, BERT_80M), (LLAMA_8B, BERT_80M), (LLAMA_70B_TURBO, BERT_80M), (LLAMA_405B_TURBO, BERT_80M)]:
    # for large_model, small_model in [(GPT_4_1, BERT_80M)]:
        large_config = {INPUT_TOKENS: input_tokens, OUTPUT_TOKENS: output_tokens, MODEL_ID: large_model}
        small_config = {INPUT_TOKENS: wrapped_input_tokens, OUTPUT_TOKENS: wrapped_output_tokens, MODEL_ID: small_model}

        print()
        print(f"{small_model} vs. {large_model}")
        base_price = base_price_estimation(input_tokens, output_tokens, num_requests, large_model)
        combined_price = combined_price_estimation(large_config, small_config, num_requests, switch_after_n_items)
        print(f"base: {base_price:.2f}$")
        print(f"combined: {combined_price:.2f}$")
        print(f"difference: {base_price - combined_price:.2f}$")
        print(f"factor: {base_price / combined_price:.2f}x")

    request_range_model_pairs = [
        (GPT_4_1_NANO, [5000, 10000, 20000, 40000, 80000, 160000]),
        (GPT_4_1, [1000, 2000, 4000, 8000, 16000]),
        (LLAMA_405B_TURBO, [1000, 2000, 4000, 8000, 16000]),
    ]
    for model, requests in request_range_model_pairs:
        plot_price_savings(requests, model, BERT_80M)
