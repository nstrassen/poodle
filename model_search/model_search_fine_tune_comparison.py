import json
import os

from training.bert_training import LLM, GROUND_TRUTH


def get_max_test_accuracy_for_model(model_name, label_method, num_splits, test_splits=None):
    results_dir = "/mount-fs/poodle/fine-tune-model-results/imdb/"
    if test_splits is None:
        # sth like this rttl-ai#bert-base-uncased-yelp-reviews_results_GROUND_TRUTH_splits10_epochs10.json
        json_file_name = f"{model_name.replace('/', '#')}_results_{label_method}_splits{num_splits}_epochs10.json"
    else:
        epochs = 10 if num_splits == 1 or num_splits == 2 else 5
        json_file_name = f"{model_name.replace('/', '#')}_results_{label_method}_trainsplits{num_splits}_testsplits{test_splits}_epochs{epochs}.json"
    json_file_path = os.path.join(results_dir, json_file_name)

    # parse the json file to get the max test accuracy
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        accuracies = [epoch_data["test_accuracy"] for epoch_data in data["epochs"]]
        max_test_accuracy = max(accuracies)
        return max_test_accuracy


if __name__ == '__main__':
    max_test_accuracy = {}
    for model_name in [
        "google-bert/bert-base-uncased",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "rttl-ai/bert-base-uncased-yelp-reviews",
        "saitejautpala/bert-base-yelp-reviews",
        "Elenapervova/bio-distilbert-uncased-mimic-iii",
        "textattack/bert-base-uncased-CoLA",
        "cnut1648/biolinkbert-mednli",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        "JungleLee/bert-toxic-comment-classification",
        "vittoriomaggio/bert-base-msmarco-fiqa",
    ]:
        # max_test_accuracy[model_name] = get_max_test_accuracy_for_model(model_name, LLM, num_splits=10)
        max_test_accuracy[model_name] = get_max_test_accuracy_for_model(model_name, GROUND_TRUTH, num_splits=10)

    # sort by max test accuracy
    sorted_max_test_accuracy = dict(sorted(max_test_accuracy.items(), key=lambda item: item[1], reverse=True))
    print("Max test accuracy for models (sorted):")
    for model_name, accuracy in sorted_max_test_accuracy.items():
        print(f"{model_name}: {accuracy}")

    # GT labels, splits 1
    # Max test accuracy for models (sorted):
    # rttl-ai/bert-base-uncased-yelp-reviews: 0.936
    # distilbert/distilbert-base-uncased-finetuned-sst-2-english: 0.898
    # saitejautpala/bert-base-yelp-reviews: 0.896
    # google-bert/bert-base-uncased: 0.874
    # Hate-speech-CNERG/bert-base-uncased-hatexplain: 0.85
    # vittoriomaggio/bert-base-msmarco-fiqa: 0.806
    # cnut1648/biolinkbert-mednli: 0.792
    # textattack/bert-base-uncased-CoLA: 0.764
    # JungleLee/bert-toxic-comment-classification: 0.62
    # Elenapervova/bio-distilbert-uncased-mimic-iii: 0.598

    model_test_accuracies = {}
    for model_name in [
        "google-bert/bert-base-uncased",
        "rttl-ai/bert-base-uncased-yelp-reviews",
    ]:
        model_test_accuracies[model_name] = {}
        for split_count in [1, 2, 4, 10]:
            model_test_accuracies[model_name][split_count * 500] = \
                get_max_test_accuracy_for_model(model_name, LLM, num_splits=split_count, test_splits=10)

    print("\nMax test accuracy for models by number of training samples:")
    for model_name, accuracy in model_test_accuracies.items():
        print(f"{model_name}:")
        for num_train_samples, acc in accuracy.items():
            print(f"  {num_train_samples} samples: {acc}")

#     Max test accuracy for models by number of training samples:
# google-bert/bert-base-uncased:
#   500 samples: 0.8662
#   1000 samples: 0.8758
#   2000 samples: 0.8796
#   5000 samples: 0.8932
# rttl-ai/bert-base-uncased-yelp-reviews:
#   500 samples: 0.9074
#   1000 samples: 0.9096
#   2000 samples: 0.9146
#   5000 samples: 0.9182
