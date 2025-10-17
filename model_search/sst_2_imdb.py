import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from training.bert_training import load_imdb_splits, IMDbDataset


def evaluate_sst2_on_imdb(sst_model, tokenizer, root_data_path, batch_size=8, max_length=512, num_splits=1,
                          mapping_function=None):
    # ----- 1. Load IMDB test data -----
    _, _, test_texts, test_labels = load_imdb_splits(root_data_path, num_splits, label_method="GROUND_TRUTH")

    # ----- 2. Tokenize -----
    encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encodings["labels"] = torch.tensor(test_labels)

    test_dataset = IMDbDataset(encodings)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ----- 4. Evaluate -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sst_model.to(device)
    sst_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = sst_model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            if mapping_function:
                predictions = mapping_function(predictions)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def map_yelp_stars_to_imdb(star_ratings):
    """
    Maps Yelp star ratings (1–5) to SST-2 binary sentiment labels.
      - 1–2 stars → 0 (negative)
      - 3–5 stars → 1 (positive)
    """
    if not isinstance(star_ratings, torch.Tensor):
        star_ratings = torch.tensor(star_ratings)

    # 1–2 stars → 0, 3–5 stars → 1
    mapped = torch.where(star_ratings <= 2, 0, 1)
    return mapped


if __name__ == '__main__':
    results = {}
    for model_name, mapping_func in [
        ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", None),
        ("rttl-ai/bert-base-uncased-yelp-reviews", map_yelp_stars_to_imdb),
        ("saitejautpala/bert-base-yelp-reviews", map_yelp_stars_to_imdb),
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        root_data_path = f"/mount-fs/poodle/labeled-data/imdb/"
        batch_size = 32
        num_splits = 10  # 5000 items


        results[model_name] = evaluate_sst2_on_imdb(model, tokenizer, root_data_path, batch_size, 512, num_splits,
                                                    mapping_function=mapping_func)
    output_path = "imdb_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")
    print(results)

#     results
# {'distilbert/distilbert-base-uncased-finetuned-sst-2-english': 0.8942, 'rttl-ai/bert-base-uncased-yelp-reviews': 0.8932, 'saitejautpala/bert-base-yelp-reviews': 0.8644}

