import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler

from data.imdb.reduced_imdb import load_imdb_data

if __name__ == '__main__':
    # TODO write logic to combine splits as well
    root_data_path = '/tmp/pycharm_project_761/data'
    # ----- 1. Load your IMDB data -----
    train_path = os.path.join(root_data_path, "imdb/data/aclImdb/train-500-0")  # update this path
    test_path = os.path.join(root_data_path, "imdb/data/aclImdb/test-500-0")

    train_texts, train_labels = load_imdb_data(train_path)
    test_texts, test_labels = load_imdb_data(test_path)

    # ----- 2. Tokenize data -----
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def tokenize_texts(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
        encodings['labels'] = torch.tensor(labels)
        return encodings


    train_encodings = tokenize_texts(train_texts, train_labels)
    test_encodings = tokenize_texts(test_texts, test_labels)


    # ----- 3. Build a Dataset class -----
    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}


    train_dataset = IMDbDataset(train_encodings)
    test_dataset = IMDbDataset(test_encodings)

    # ----- 4. Create DataLoaders -----
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # ----- 5. Load pretrained BERT -----
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- 6. Optimizer & Scheduler -----
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
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
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

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

    # ----- 9. Save fine-tuned model -----
    save_path = "./bert-imdb-finetuned"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
