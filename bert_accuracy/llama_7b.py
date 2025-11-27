import json
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.imdb.reduced_imdb import load_imdb_data


def load_llama_model(model_name: str):
    """Load LLaMA tokenizer and model with memory-efficient settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tokenizer, model


def classify_review(tokenizer, model, review_text: str, labels=["positive", "negative"]):
    """Zero-shot sentiment classification using LLaMA."""
    prompt = f"Classify the sentiment of the following movie review as positive or negative:\n\n{review_text}\n\nSentiment:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace(prompt, "")

    for label in labels:
        if label.lower() in generated_text.lower():
            return label
    return "neutral"


def llm_labeled_dataset(tokenizer, model, texts, labels, file_names):
    llm_labels = []
    misclassified = []
    for review, true_label, file_name in tqdm(zip(texts, labels, file_names), total=len(texts)):
        pred_label = classify_review(tokenizer, model, review)
        pred = 1 if pred_label == "positive" else 0
        llm_prediction = {"file": file_name, "true_label": true_label, "llm_label": pred}
        llm_labels.append(llm_prediction)
        if pred != true_label:
            misclassified.append(llm_prediction)
    return llm_labels, misclassified


def write_labels_to_file(llm_labels, misclassified, model_name, save_dir):
    json_dict = {
        "llm_model": model_name,
        "llm_predictions": llm_labels,
        "misclassified_items": misclassified
    }
    with open(os.path.join(save_dir, "llm_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


def main():
    # --- 1. Load LLaMA 7B ---
    model_name = "NousResearch/Llama-2-7b-chat-hf"  # replace with a public accessible variant
    tokenizer, model = load_llama_model(model_name)

    # --- 2. Load IMDB test data ---
    for split in ["train", "test"]:
        for partition in range(50):
            root_data_path = '/tmp/pycharm_project_761/data'
            test_path = os.path.join(root_data_path, f"imdb/data/aclImdb/{split}-500-{partition}")
            test_texts, test_labels, file_names = load_imdb_data(test_path, include_file_names=True)

            # --- 3. Generate and save LLama labels ---
            save_dir = os.path.join(test_path)
            llm_labels, misclassified = llm_labeled_dataset(tokenizer, model, test_texts, test_labels, file_names)

            write_labels_to_file(llm_labels, misclassified, model_name, save_dir)


if __name__ == "__main__":
    main()
