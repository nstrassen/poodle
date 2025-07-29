import os

import openai
from together import Together

OPEN_AI_MODELS = ["gpt-3.5-turbo", "gpt-4"]
TOGETHER_MODELS = ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]

WRAPPER_PROMPT = """
You are a powerful language model. You are given a user request consisting of a system prompt and a user message.
Your task is as follows:

1. Identify the type of input the user is providing: one of ["plain text", "JSON", "image", "table", "other"]
2. Infer what task you are expected to perform, choosing from: 
   ["sentiment classification", "summarization", "translation", "question answering", "information extraction", "topic modeling", "other"]
3. Solve the task that the user is giving you and put your response in a separate field called "user-response" which can be in JSON format

Respond with a JSON object containing:
- "input_type"
- "task_type"
- "user-response" (JSON describing user response)
"""


def openAI_classify_review(system_prompt, review_text, model_name, collect_statistics=True):
    messages = [
        {"role": "system", "content": system_prompt},
        dict(role="user", content=f"REVIEW: {review_text}")
    ]

    if collect_statistics:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": WRAPPER_PROMPT},
                dict(role="user", content=f"USER REQUEST: {str(messages)}")
            ],
            temperature=0,
            # response_format="json" only works with latest API and most recent models
        )

    else:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
            # response_format="json" only works with latest API and most recent models
        )

    raw_output = response['choices'][0]['message']['content'].strip()
    return raw_output


def together_classify_review(system_prompt, review_text, model_name, collect_statistics=True):
    messages = [
        {"role": "system", "content": system_prompt},
        dict(role="user", content=f"REVIEW: {review_text}")
    ]

    client = Together()

    if collect_statistics:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": WRAPPER_PROMPT},
                dict(role="user", content=f"USER REQUEST: {str(messages)}")
            ],
            temperature=0,
        )
    else:
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

    review_text = "I was completely bored the entire time. The acting was flat and the plot was nonsensical."

    SYSTEM_PROMPT = (
        "You are a sentiment analysis assistant. "
        "Your job is to read movie reviews and classify their sentiment as either 'positive' or 'negative'. "
        "Only respond in this exact JSON format: {\"sentiment\": \"positive\"} or {\"sentiment\": \"negative\"}. "
        "Do not provide any explanation or additional text."
    )
    #
    # response = classify_review(SYSTEM_PROMPT, review_text, "gpt-3.5-turbo", collect_statistics=False)
    # print(response)
    #

    response = classify_review(SYSTEM_PROMPT, review_text, "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                               collect_statistics=True)
    print(response)
