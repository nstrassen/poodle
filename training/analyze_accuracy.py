import json
import os.path
from statistics import mean


def analyze_file(file_path):
    # Open and load JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    predictions = data.get("llm_predictions", [])
    if not predictions:
        print("No predictions found in the file.")
        return None

    correct = sum(1 for p in predictions if p["true_label"] == p["llm_label"])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    return accuracy

def analyze_accuracy_per_file(files):
    result = {}
    for f in files:
        result[f] = analyze_file(f)

    return result


if __name__ == '__main__':
    for split in ["train", "test"]:
        root_train_path = f"/mount-fs/poodle/labeled-data/imdb/{split}-500-splits"
        train_paths = [os.path.join(root_train_path, f"{split}-500-{i}", "llm_predictions.json") for i in range(50)]
        train_results = analyze_accuracy_per_file(train_paths)
        print(f"{split} results", train_results)
        print(f"avg {split} accuracy", mean(list(train_results.values())))
