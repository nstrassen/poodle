import torch
from torch.utils.data import DataLoader


# code from Alsatian paper

def get_input_dimension(batch):
    sample_tensor: torch.Tensor = batch[0]
    return sample_tensor.shape


# python
# File: model_search/nn_proxy.py
import torch
from torch.utils.data import DataLoader, TensorDataset


def linear_proxy(train_data=None, test_data=None, num_classes=None, device: torch.device = None,
                 batch_size: int = 32, epochs: int = 100) -> (float, float):
    """
    train_data / test_data must be tuples: (features_tensor, labels_tensor)
    features: (N, D), labels: (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dataloader_from_tensors(feat: torch.Tensor, lab: torch.Tensor, bs: int, shuffle: bool):
        feat = feat.to(device)
        lab = lab.to(device)
        if lab.dtype != torch.long:
            lab = lab.long()
        ds = TensorDataset(feat, lab)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    if not (isinstance(train_data, tuple) and isinstance(test_data, tuple)):
        raise ValueError("train_data and test_data must be tuples: (features_tensor, labels_tensor)")

    train_features, train_labels = train_data
    test_features, test_labels = test_data

    input_dim = train_features.shape[1]
    if num_classes is None:
        num_classes = int(train_labels.max().item()) + 1

    train_loader = to_dataloader_from_tensors(train_features, train_labels, batch_size, shuffle=True)
    test_loader = to_dataloader_from_tensors(test_features, test_labels, batch_size, shuffle=False)

    model = torch.nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for feature_batch, label_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(feature_batch)
            loss = loss_func(outputs, label_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for feature_batch, label_batch in test_loader:
            outputs = model(feature_batch)
            loss = loss_func(outputs, label_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += label_batch.size(0)
            correct_predictions += (predicted == label_batch).sum().item()

    average_loss = (total_loss / len(test_loader)) if len(test_loader) > 0 else 0.0
    top1_accuracy = (correct_predictions / total_samples) if total_samples > 0 else 0.0

    return {"loss": average_loss, "accuracy": top1_accuracy}


def collect_features_and_labels(caching_service, device, train_feature_ids, train_label_ids, bert_features):
    features = []
    labels = []
    for feature_id, label_id in zip(train_feature_ids, train_label_ids):
        feature_batch = caching_service.get_item(feature_id)
        label_batch = caching_service.get_item(label_id)
        if bert_features:
            last_hidden_state = feature_batch[0]
            cls_hidden_state = last_hidden_state[:, 0, :]
            feature_batch = cls_hidden_state
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        feature_batch, label_batch = torch.squeeze(feature_batch), torch.squeeze(label_batch)
        features.append(feature_batch)
        labels.append(label_batch)
    return features, labels
