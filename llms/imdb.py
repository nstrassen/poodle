import os

import openai
from together import Together

OPEN_AI_MODELS = ["gpt-3.5-turbo", "gpt-4"]
TOGETHER_MODELS = ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]

WRAPPER_PROMPT = """
You are a powerful language model. You are given a user request consisting of a system prompt and a user message.
Your task is as follows:

1. Identify the type of input the user is providing: one of ["text", "image", "table", "other"]
2. Infer what task you are expected to perform, choosing from: 
   ["sentiment classification", "summarization", "translation", "question answering", "information extraction", "topic modeling", "other"]
3. Solve the task that the user is giving you and put your response in a separate field called "user-response" which can be in JSON format

Respond with a JSON object containing:
- "input_type"
- "task_type"
- "user-response" (JSON describing user response)
"""


def _base_messages(review_text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"REVIEW: {review_text}"}
    ]
    return messages


def _wrapped_messages(messages):
    messages = [
        {"role": "system", "content": WRAPPER_PROMPT},
        {"role": "user", "content": f"USER REQUEST: {str(messages)}"}
    ]
    return messages


def openAI_classify_review(system_prompt, review_text, model_name, collect_statistics=True):
    messages = _base_messages(review_text, system_prompt)

    if collect_statistics:
        messages = _wrapped_messages(messages)

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        # response_format="json" only works with latest API and most recent models
    )
    raw_output = response['choices'][0]['message']['content'].strip()
    return raw_output


def together_classify_review(system_prompt, review_text, model_name, collect_statistics=True):
    client = Together()

    messages = _base_messages(review_text, system_prompt)

    if collect_statistics:
        messages = _wrapped_messages(messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()


def classify_review(system_prompt, review_text, model_name, collect_statistics=True):
    if model_name in OPEN_AI_MODELS:
        return openAI_classify_review(system_prompt, review_text, model_name, collect_statistics=collect_statistics)
    else:
        return together_classify_review(system_prompt, review_text, model_name, collect_statistics=collect_statistics)


def together_test():
    client = Together()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "user",
                "content": "What is 3+4, reply in one word"
            }
        ]
    )
    print(response.choices[0].message.content)


if __name__ == '__main__':
    api_key = os.getenv("TOGETHER_API_KEY")
    print(api_key)

    # example of a review classification similar to IMDB
    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Your job is to read movie reviews and classify their sentiment as either 'positive' or 'negative'. "
        "Only respond in this exact JSON format: {\"sentiment\": \"positive\"} or {\"sentiment\": \"negative\"}. "
        "Do not provide any explanation or additional text."
    )
    review_text = "I was completely bored the entire time. The acting was flat and the plot was nonsensical."

    # example of how to use openAI (currently we do not have an API key, but can try the code in a Notebook in Coursera course)
    # https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/tyucw/inferring
    # model_name = "gpt-3.5-turbo"
    # response = classify_review(system_prompt, review_text, model_name, collect_statistics=True)
    # print(f"RESPONSE FROM OpenAI ({model_name}):\n {response}")

    # chose this model because currently it seems to be free
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    response = classify_review(system_prompt, review_text, model_name, collect_statistics=True)
    print(f"RESPONSE FROM TogetherAI ({model_name}):\n {response}")
