import openai

OPEN_AI_MODELS = ["gpt-3.5-turbo", "gpt-4"]


def openAI_classify_review(system_prompt, review_text, model_name):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            dict(role="user", content=f"REVIEW: {review_text}")
        ],
        temperature=0,
        # response_format="json" only works with latest API and most recent models
    )

    raw_output = response['choices'][0]['message']['content'].strip()
    return raw_output


def classify_review(system_prompt, review_text, model_name):
    if model_name in OPEN_AI_MODELS:
        return openAI_classify_review(system_prompt, review_text, model_name)


if __name__ == '__main__':
    review_text = "I was completely bored the entire time. The acting was flat and the plot was nonsensical."

    SYSTEM_PROMPT = (
        "You are a sentiment analysis assistant. "
        "Your job is to read movie reviews and classify their sentiment as either 'positive' or 'negative'. "
        "Only respond in this exact JSON format: {\"sentiment\": \"positive\"} or {\"sentiment\": \"negative\"}. "
        "Do not provide any explanation or additional text."
    )

    response = classify_review(SYSTEM_PROMPT, review_text, "gpt-3.5-turbo")
    print(response)

