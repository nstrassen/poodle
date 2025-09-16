import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from data.imdb.reduced_imdb import get_imbdb_bert_base_uncased_datasets
from models.sequential_bert import get_sequential_bert_model, BertLinearHead
from util.costants import TRAIN, TEST


# Fine-tuning function
def fine_tune_model(model, train_data, epochs=10, lr=10e-4, val_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Freeze all parameters except the final linear head
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            print("batch")
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model([input_ids, attention_mask])
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = logits.max(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss:.4f}")
        train_accuracy = correct_train / total_train
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model([input_ids, attention_mask])
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        model.train()


if __name__ == '__main__':
    dataset_paths = {
        TRAIN: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train-1000",
        TEST: "/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test-1000"
    }

    train_data = get_imbdb_bert_base_uncased_datasets(dataset_paths[TRAIN])
    test_data = get_imbdb_bert_base_uncased_datasets(dataset_paths[TEST])

    model = get_sequential_bert_model(pretrained=True)

    model: torch.nn.Sequential = model
    model.append(
        BertLinearHead()
    )

    fine_tune_model(model, train_data)
