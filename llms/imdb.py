import os
import time

from data.imdb.reduced_imdb import get_plain_imdb_data
from llms.api_call_and_prompts import classify_review
from util.costants import *
from util.files import save_dict_to_json, load_dict_from_json

dataset_paths = {
    TRAIN: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train",
    TRAIN_100: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train-100",
    TRAIN_1000: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train-1000",
    TEST: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test",
    TEST_100: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test-100",
    TEST_1000: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test-1000"
}


def imdb_100_inference():


    train_texts, train_labels = get_plain_imdb_data(dataset_paths[TRAIN])

    print("test")


def imdb_1000_statistics():

    train_texts, train_labels = get_plain_imdb_data(dataset_paths[TRAIN_1000])
    test_texts, test_labels = get_plain_imdb_data(dataset_paths[TEST_1000])

    calc_avg_word_count(train_texts)
    calc_avg_word_count(test_texts)


def calc_avg_word_count(train_texts):
    word_count = 0
    for text in train_texts:
        word_count += len(text.split())
    avg_word_count = word_count / len(train_texts)
    print("avg word count: ", avg_word_count)
    return avg_word_count

def get_average_sample():
    train_texts, train_labels = get_plain_imdb_data(dataset_paths[TRAIN_1000])
    avg_word_count = calc_avg_word_count(train_texts)
    for text, label in zip(train_texts, train_labels):
        word_count = len(text.split())
        print(word_count)
        if word_count - 5 < avg_word_count < word_count + 5:
            return text, label

def trigger_together_ai_comparison(text):
    api_key = os.getenv("TOGETHER_API_KEY")
    print(api_key)

    # example of a review classification similar to IMDB
    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Your job is to read movie reviews and classify their sentiment as either 'positive' or 'negative'. "
        "Only respond in this exact JSON format: {\"sentiment\": \"positive\"} or {\"sentiment\": \"negative\"}. "
        "Do not provide any explanation or additional text."
    )
    review_text = text

    # chose this model because currently it seems to be free
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


    # first inference without wrapper prompt for statistics collection
    response = classify_review(system_prompt, review_text, model_name, wrapper_prompt=False)
    save_dict_to_json(response, "non-wrapped-response.json")
    print(f"RESPONSE FROM TogetherAI ({model_name}):\n {response[RAW_OUTPUT]}")

    # sleep for 2 min to have gap in dashboard
    time.sleep(120)

    # second inference with wrapper prompt for statistics collection
    response = classify_review(system_prompt, review_text, model_name, wrapper_prompt=True)
    print(f"RESPONSE FROM TogetherAI ({model_name}):\n {response[RAW_OUTPUT]}")
    save_dict_to_json(response, "wrapped-response.json")

def analysis():
    non_wrapped = load_dict_from_json("non-wrapped-response.json")
    wrapped = load_dict_from_json("wrapped-response.json")

    TOKEN_WORD_FAC = 1.3

    print("NON WRAPPED")
    print(f"approximated tokens:\n{int(len(str(non_wrapped[SENT_MESSAGES]).split()) * TOKEN_WORD_FAC)}")

    print("WRAPPED")
    print(f"approximated tokens:\n{int(len(str(wrapped[SENT_MESSAGES]).split()) * TOKEN_WORD_FAC)}")



if __name__ == '__main__':
    # imdb_1000_statistics()
    # text, label = get_average_sample()
    # print(text)
    # print(label)
    # trigger_together_ai_comparison(text)

    analysis()