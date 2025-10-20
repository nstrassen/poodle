import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model_search.nn_proxy import linear_proxy
from training.bert_training import load_imdb_splits, IMDbDataset


def _get_data_loader(data, batch_size, max_length, tokenizer, split):
    encodings = tokenizer(data[split]["text"], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
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

def score_models(data, model_ids, num_classes, batch_size=32, max_length=512):
    features_labels = {}
    model_scores = {}
    for model_id in model_ids:
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

        # pass precomputed tensors to linear_proxy (it will infer num_classes if None)
        model_scores[model_id] = linear_proxy(
            train_data=(train_features, train_labels),
            test_data=(test_features, test_labels),
            num_classes=num_classes,
            device=device,
            batch_size=32,
            epochs=100
        )

    # sort by accuracy (higher is better)
    sorted_model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1]["accuracy"], reverse=True))
    return sorted_model_scores




if __name__ == '__main__':
    results = {}
    # TODO test more models
    model_ids = [
        "google-bert/bert-base-uncased",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "rttl-ai/bert-base-uncased-yelp-reviews",
        # "saitejautpala/bert-base-yelp-reviews",
        # "Elenapervova/bio-distilbert-uncased-mimic-iii",
        # "textattack/bert-base-uncased-CoLA",
        # "cnut1648/biolinkbert-mednli",
        # "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        # "JungleLee/bert-toxic-comment-classification",
        # "vittoriomaggio/bert-base-msmarco-fiqa",
    ]
    root_data_path = f"/mount-fs/poodle/labeled-data/imdb/"
    num_splits = 1  # 1 split 500 items
    train_texts, train_labels, test_texts, test_labels =\
        load_imdb_splits(root_data_path, num_splits, label_method="GROUND_TRUTH")
    data = {
        "train": {"text": train_texts, "labels": train_labels},
        "test": {"text": test_texts, "labels": test_labels}
    }

    model_scores = score_models(data, model_ids, num_classes=2, batch_size=32, max_length=256)
    print(model_scores)



    # TODO fine-tune all models


