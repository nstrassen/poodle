import os

from bert_accuracy.bert_training import BERT_BASE_UNCASED, train_and_evaluate_bert_like_model, GROUND_TRUTH, LLM

if __name__ == '__main__':
    results_dir = "/mount-fs/poodle/fine-tune-model-results/imdb/"
    os.makedirs(results_dir, exist_ok=True)
    root_data_path = f"/mount-fs/poodle/labeled-data/imdb/"
    model_name = BERT_BASE_UNCASED
    batch_size = 32

    # model_names = [
    #     "google-bert/bert-base-uncased",
    #     "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    #     "rttl-ai/bert-base-uncased-yelp-reviews",
    #     "saitejautpala/bert-base-yelp-reviews",
    #     "Elenapervova/bio-distilbert-uncased-mimic-iii",
    #     "textattack/bert-base-uncased-CoLA",
    #     "cnut1648/biolinkbert-mednli",
    #     "Hate-speech-CNERG/bert-base-uncased-hatexplain",
    #     "JungleLee/bert-toxic-comment-classification",
    #     "vittoriomaggio/bert-base-msmarco-fiqa",
    # ]

    model_names = [
        "google-bert/bert-base-uncased",
        "rttl-ai/bert-base-uncased-yelp-reviews",
    ]
    for model_name in model_names:
        for number_of_splits, num_epochs in zip([1, 2, 4, 10], [10, 10, 5, 5]):
            for label_method in [GROUND_TRUTH, LLM]:
                print(f"Fine-tuning and evaluating model {model_name} with {number_of_splits} splits...")
                train_and_evaluate_bert_like_model(
                    root_data_path, number_of_train_splits=number_of_splits, number_of_test_splits=10,
                    num_epochs=num_epochs, batch_size=batch_size, model_name=model_name, results_dir=results_dir,
                    label_method=label_method)
