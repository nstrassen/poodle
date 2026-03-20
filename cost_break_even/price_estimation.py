from demo.demo_config import DemoScenario, Model
from demo.token_count_estimation import num_tokens
from demo.x_values import get_plot_x_ticks

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
    Model.BERT_80M: {INPUT: 0.01, OUTPUT: 0.01},
    # Ref
    Model.LLAMA_8B: {INPUT: 0.20, OUTPUT: 0.02},
    # https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo
    Model.LLAMA_70B_TURBO: {INPUT: 0.88, OUTPUT: 0.88},
    # https://api.together.ai/models/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
    Model.LLAMA_405B_TURBO: {INPUT: 3.50, OUTPUT: 3.50},
    # https://platform.openai.com/docs/pricing
    # - gpt 4.1: 2.00\$ (input), 8.00\$ (output), batch prices half
    Model.GPT_4_1: {INPUT: 2, OUTPUT: 8},
    # - gpt 4.1-mini: 0.40\$ (input), 1.60\$ (output), batch prices half
    Model.GPT_4_1_MINI: {INPUT: 0.4, OUTPUT: 1.6},
    # - gpt 4.1-nano: 0.10\$ (input), 0.40\$ (output), batch prices half
    Model.GPT_4_1_NANO: {INPUT: 0.1, OUTPUT: 0.4},
}


def demo_price_estimation(config: DemoScenario):
    large_model_costs = single_model_price(config.models.large_model, config)


def single_model_price(model_id, config: DemoScenario):
    price_per_request = single_model_price_per_request(model_id, config)
    price = price_per_request * config.requests.expected_requests

    return price


def single_model_price_per_request(model_id, config, wrapped=False):
    inp_token_price = MODEL_PRICING_PER_1M[model_id][INPUT] / 10 ** 6
    out_token_price = MODEL_PRICING_PER_1M[model_id][OUTPUT] / 10 ** 6

    input_tokens = num_tokens(config.tokens.input) + num_tokens(config.tokens.prompt)
    if wrapped:
        input_tokens += num_tokens(config.tokens.wrapper_prompt)

    if wrapped:
        output_tokens = num_tokens(config.tokens.wrapped_output)
    else:
        output_tokens = num_tokens(config.tokens.output)

    price_per_request = inp_token_price * input_tokens + out_token_price * output_tokens

    return price_per_request


def poodle_price(config: DemoScenario):
    large_model_price_per_request_wrapped = \
        single_model_price_per_request(config.models.large_model, config, wrapped=True)
    large_model_price_per_request_non_wrapped = \
        single_model_price_per_request(config.models.large_model, config, wrapped=False)
    small_model_price_per_request = \
        single_model_price_per_request(config.models.small_model, config, wrapped=False)

    # first n request that go to lagre model
    large_model_requests = min(config.requests.expected_requests, config.requests.switch_after_n_items)
    large_model_requests_wrapped = config.tokens.wrapped_requests_percent * large_model_requests
    large_model_requests_non_wrapped = large_model_requests - large_model_requests_wrapped
    first_n_cost = (large_model_requests_non_wrapped * large_model_price_per_request_non_wrapped +
                    large_model_requests_wrapped * large_model_price_per_request_wrapped)

    # next requests to small model
    small_model_requests = max(config.requests.expected_requests - config.requests.switch_after_n_items, 0)
    small_model_cost = small_model_requests * small_model_price_per_request

    # every now and then send request to large model to check for data drift
    monitoring_requests = config.validation.validation_requests_percent * small_model_requests
    monitoring_cost = monitoring_requests * large_model_price_per_request_non_wrapped


    # TODO get better numbers here, probably cost will go lower
    # for now assume GPU time of 3 hours for model search and fine-tuning, have to validate
    # cost per hour ~1.2$/h -> ~4$
    model_search_and_dev_cost = config.dev.model_dev_costs

    return first_n_cost + small_model_cost + monitoring_cost + model_search_and_dev_cost


def compare_single_model_and_poodle(config: DemoScenario):
    request_values = get_plot_x_ticks(config.requests.expected_requests)
    base_prices = {}
    poodle_prices = {}
    poodle_savings = {}
    for request in request_values:
        config.requests.expected_requests = request
        base_prices[request] = single_model_price(config.models.large_model, config)
        poodle_prices[request] = poodle_price(config)
        poodle_savings[request] = base_prices[request] - poodle_prices[request]

    return base_prices, poodle_prices, poodle_savings

if __name__ == '__main__':
    example_scenario = DemoScenario.get_example_scenario()
    base_prices, poodle_prices, poodle_savings = compare_single_model_and_poodle(example_scenario)

    print("Base prices:", base_prices)
    print("Poodle prices:", poodle_prices)
    print("Poodle savings:", poodle_savings)