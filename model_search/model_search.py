def _get_data_loader(data, batch_size, max_length, tokenizer, split):
    encodings = tokenizer(data[split]["text"], truncation=True, padding=True, max_length=max_length,
                          return_tensors="pt")
    encodings["labels"] = torch.tensor(data[split]["labels"])
    dataset = IMDbDataset(encodings)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def _extract_features(loader, feature_model, device):
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = feature_model(**inputs)

            # Prefer pooler_output if available
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                # fallback: use [CLS] token embedding
                feat = outputs.last_hidden_state[:, 0, :]

            all_feats.append(feat.cpu())
            all_labels.append(batch["labels"].cpu())

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


# python
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer

from model_search.nn_proxy import linear_proxy
from training.bert_training import load_imdb_splits, IMDbDataset


def score_models(data, model_ids, num_classes, batch_size=32, max_length=512):
    features_labels = {}
    model_scores = {}

    overall_start = time.perf_counter()
    for model_id in model_ids:
        model_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        feature_model = AutoModel.from_pretrained(model_id)
        for param in feature_model.parameters():
            param.requires_grad = False

        train_loader = _get_data_loader(data, batch_size, max_length, tokenizer, "train")
        test_loader = _get_data_loader(data, batch_size, max_length, tokenizer, "test")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_model.to(device)
        feature_model.eval()

        train_features, train_labels = _extract_features(train_loader, feature_model, device)
        test_features, test_labels = _extract_features(test_loader, feature_model, device)

        features_labels[model_id] = {"train": (train_features, train_labels), "test": (test_features, test_labels)}

        # time the scoring of this model
        model_scores[model_id] = linear_proxy(
            train_data=(train_features, train_labels),
            test_data=(test_features, test_labels),
            num_classes=num_classes,
            device=device,
            batch_size=32,
            epochs=100
        )
        model_end = time.perf_counter()
        elapsed = model_end - model_start

        # store loss, accuracy and per-model time
        model_scores[model_id]["score-time"] = elapsed

    overall_end = time.perf_counter()
    total_time = overall_end - overall_start

    # sort by accuracy (higher is better)
    sorted_model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1]["accuracy"], reverse=True))

    return sorted_model_scores, total_time


if __name__ == '__main__':
    results = {}
    model_ids = [
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
    ]
    root_data_path = f"/mount-fs/poodle/labeled-data/imdb/"
    num_splits = 1  # 1 split 500 items
    train_texts, train_labels, test_texts, test_labels = \
        load_imdb_splits(root_data_path, num_splits, label_method="GROUND_TRUTH")
    data = {
        "train": {"text": train_texts, "labels": train_labels},
        "test": {"text": test_texts, "labels": test_labels}
    }

    model_scores, scoring_time = score_models(data, model_ids, num_classes=2, batch_size=32, max_length=256)
    print(f"model_scores:")
    for k, v in model_scores.items():
        print(f"  {k}: {v}")
    print(f"scoring_time: {scoring_time}")
    # model_scores:
    #   rttl-ai/bert-base-uncased-yelp-reviews: {'loss': 0.20674960769247264, 'accuracy': 0.916, 'score-time': 5.533341416157782}
    #   distilbert/distilbert-base-uncased-finetuned-sst-2-english: {'loss': 0.3754851398989558, 'accuracy': 0.89, 'score-time': 3.401031568646431}
    #   saitejautpala/bert-base-yelp-reviews: {'loss': 0.33620492462068796, 'accuracy': 0.882, 'score-time': 5.402611096389592}
    #   textattack/bert-base-uncased-CoLA: {'loss': 0.8060485599562526, 'accuracy': 0.76, 'score-time': 5.682511903345585}
    #   JungleLee/bert-toxic-comment-classification: {'loss': 1.0297829303890467, 'accuracy': 0.728, 'score-time': 5.665892992168665}
    #   cnut1648/biolinkbert-mednli: {'loss': 5.833794282509393, 'accuracy': 0.574, 'score-time': 15.262184121645987}
    #   Hate-speech-CNERG/bert-base-uncased-hatexplain: {'loss': 6.493016869611779, 'accuracy': 0.532, 'score-time': 6.066282565705478}
    #   google-bert/bert-base-uncased: {'loss': 4.9824152248038445, 'accuracy': 0.514, 'score-time': 5.9154657907783985}
    #   vittoriomaggio/bert-base-msmarco-fiqa: {'loss': 14.546208537002755, 'accuracy': 0.504, 'score-time': 5.652437528595328}
    #   Elenapervova/bio-distilbert-uncased-mimic-iii: {'loss': 6.356876552070105, 'accuracy': 0.5, 'score-time': 5.39708112180233}
    # scoring_time: 63.978852627798915

    # model_scores:
    #   rttl-ai/bert-base-uncased-yelp-reviews: {'loss': 0.1956797163002193, 'accuracy': 0.928, 'score-time': 5.563915494829416}
    #   distilbert/distilbert-base-uncased-finetuned-sst-2-english: {'loss': 0.3654198064468801, 'accuracy': 0.88, 'score-time': 3.3944723177701235}
    #   saitejautpala/bert-base-yelp-reviews: {'loss': 0.4343876102939248, 'accuracy': 0.8, 'score-time': 5.446939000859857}
    #   cnut1648/biolinkbert-mednli: {'loss': 1.7444217577576637, 'accuracy': 0.772, 'score-time': 15.301548249088228}
    #   google-bert/bert-base-uncased: {'loss': 1.0629445556551218, 'accuracy': 0.74, 'score-time': 5.952922770753503}
    #   Hate-speech-CNERG/bert-base-uncased-hatexplain: {'loss': 1.5466033555567265, 'accuracy': 0.712, 'score-time': 6.060503216460347}
    #   JungleLee/bert-toxic-comment-classification: {'loss': 2.4948720547836274, 'accuracy': 0.698, 'score-time': 5.706752770580351}
    #   textattack/bert-base-uncased-CoLA: {'loss': 2.7680087983753765, 'accuracy': 0.568, 'score-time': 5.7187109449878335}
    #   vittoriomaggio/bert-base-msmarco-fiqa: {'loss': 7.06517022034118, 'accuracy': 0.52, 'score-time': 5.702336234971881}
    #   Elenapervova/bio-distilbert-uncased-mimic-iii: {'loss': 4.558863144336101, 'accuracy': 0.5, 'score-time': 5.415759056806564}
    # scoring_time: 64.26387006789446

    # TODO fine-tune all models
