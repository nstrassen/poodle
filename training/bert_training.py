import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler

from data.imdb.reduced_imdb import load_imdb_data

GROUND_TRUTH = "GROUND_TRUTH"
LLM = "LLM"


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def load_imdb_splits(root_data_path, number_of_splits, label_method):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for split_idx in range(number_of_splits):
        for split_type in ["train", "test"]:
            data_path = os.path.join(root_data_path, f"{split_type}-500-splits", f"{split_type}-500-{split_idx}")
            if label_method == LLM and split_type == "train":
                texts, labels = load_imdb_data(data_path, use_llm_labels=True)
            else:
                texts, labels = load_imdb_data(data_path)
            if split_type == "train":
                train_texts.extend(texts)
                train_labels.extend(labels)
            else:
                test_texts.extend(texts)
                test_labels.extend(labels)
    return train_texts, train_labels, test_texts, test_labels


def tokenize_texts(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    encodings['labels'] = torch.tensor(labels)
    return encodings


def train_and_evaluate_bert(root_data_path, number_of_splits=10, num_epochs=3, label_method=LLM, batch_size=8,
                            model_name="bert-base-uncased"):
    # ----- 1. Load IMDB data -----
    train_texts, train_labels, test_texts, test_labels = load_imdb_splits(root_data_path, number_of_splits,
                                                                          label_method)

    # ----- 2. Tokenize data -----
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_encodings = tokenize_texts(tokenizer, train_texts, train_labels)
    test_encodings = tokenize_texts(tokenizer, test_texts, test_labels)

    # ----- 3. Split train into train/validation sets -----
    num_train = int(0.8 * len(train_encodings["input_ids"]))
    indices = torch.randperm(len(train_encodings["input_ids"]))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    # Helper function to index encodings
    def select_encodings(encodings, idxs):
        return {k: v[idxs] for k, v in encodings.items()}
    train_split_encodings = select_encodings(train_encodings, train_indices)
    val_split_encodings = select_encodings(train_encodings, val_indices)
    train_dataset = IMDbDataset(train_split_encodings)
    val_dataset = IMDbDataset(val_split_encodings)
    test_dataset = IMDbDataset(test_encodings)

    # ----- 4. Create DataLoaders -----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ----- 5. Load pretrained BERT -----
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- 6. Optimizer & Scheduler -----
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # ----- 7. Training Loop -----
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_correct += (predictions == batch["labels"]).sum().item()
            train_total += batch["labels"].size(0)

        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # ----- Validation after each epoch -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_correct += (predictions == batch["labels"]).sum().item()
                val_total += batch["labels"].size(0)
                val_steps += 1
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        model.train()

    # ----- 8. Evaluation -----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    accuracy = correct / total
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    # # ----- 9. Save fine-tuned model -----
    # save_path = "./bert-imdb-finetuned"
    # os.makedirs(save_path, exist_ok=True)
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    # print(f"Model saved to {save_path}")

    return model,accuracy


if __name__ == '__main__':
    root_data_path = f"/mount-fs/poodle/labeled-data/imdb/"
    number_of_splits = 4
    num_epochs = 3
    batch_size = 8
    model_name = "bert-base-uncased"

    result = {}
    for label_method in [LLM, GROUND_TRUTH]:
        _, accuracy = train_and_evaluate_bert(
            root_data_path, number_of_splits, num_epochs, label_method, batch_size, model_name)

        result[label_method] = accuracy

    print(result)
