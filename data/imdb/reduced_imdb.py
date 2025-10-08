import json
import os
import random
import shutil
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer

LABELS_PT = "labels.pt"

ENCODINGS_PT = "train_encodings.pt"


class CustomTensorDataset(TensorDataset):
    """
    This extends the standard PyTorch TensorDataset by adding the option to artificially size the dataset
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        super().__init__(*tensors)
        indices = torch.randperm(tensors[0].shape[0])
        shuffled_tensors = []
        for tensor in self.tensors:
            shuffled = tensor[indices]
            shuffled_tensors.append(shuffled)

        self.all_tensors = tuple(shuffled_tensors)

    def set_subrange(self, from_index, to_index):
        new_tensors = []
        for tensor in self.tensors:
            partial_tensor = tensor[from_index:to_index]
            new_tensors.append(partial_tensor)

        self.tensors = tuple(new_tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def get_llm_labels(path):
    json_path = os.path.join(path, "llm_predictions.json")
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    llm_predictions = data.get("llm_predictions", [])
    llm_labels = {prediction["file"]: prediction["llm_label"] for prediction in llm_predictions}

    return llm_labels


def load_imdb_data(path, include_file_names=False, use_llm_labels=False):
    texts = []
    labels = []
    file_names = []

    if use_llm_labels:
        llm_labels = get_llm_labels(path)

    for label in ['pos', 'neg']:
        label_path = os.path.join(path, label)
        print(f'Checking directory: {label_path}')
        if os.path.exists(label_path):
            for filename in tqdm(os.listdir(label_path), desc=f'Loading {label} reviews'):
                file_names.append(filename)
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    if use_llm_labels:
                        labels.append(llm_labels[os.path.basename(filename)])
                    else:
                        labels.append(1 if label == 'pos' else 0)
        else:
            print(f'Directory not found: {label_path}')

    if include_file_names:
        return texts, labels, file_names
    else:
        return texts, labels


def create_sub_dataset(dataset_path: str, size: int):
    new_dataset_base_path = dataset_path + f'-{size}'
    if not os.path.exists(new_dataset_base_path):
        items_per_class = int(size / 2)
        for label in ['pos', 'neg']:
            files_paths = []
            label_path = os.path.join(dataset_path, label)
            print(f'Checking directory: {label_path}')
            if os.path.exists(label_path):
                for filename in tqdm(os.listdir(label_path), desc=f'Loading {label} reviews'):
                    files_paths.append(os.path.join(label_path, filename))
            sampled_files = random.sample(files_paths, items_per_class)

            dst_path = os.path.join(new_dataset_base_path, label)
            os.makedirs(dst_path)
            for file in sampled_files:
                shutil.copy2(file, dst_path)

    return new_dataset_base_path


def create_dataset_splits(dataset_path: str, split_size: int):
    split_index = 0
    random.seed(42)  # ensure reproducibility

    all_pos_file_paths = get_all_file_paths_per_label(dataset_path, 'pos')
    all_neg_file_paths = get_all_file_paths_per_label(dataset_path, 'neg')

    random.shuffle(all_pos_file_paths)
    random.shuffle(all_neg_file_paths)

    while len(all_pos_file_paths) >= split_size and len(all_neg_file_paths) >= split_size:
        new_dataset_base_path = dataset_path + f'-{split_size * 2}-{split_index}'
        pos_split = all_pos_file_paths[:split_size]
        all_pos_file_paths = all_pos_file_paths[split_size:]
        neg_split = all_neg_file_paths[:split_size]
        all_neg_file_paths = all_neg_file_paths[split_size:]

        for label, sampled_files in zip(['pos', 'neg'], [pos_split, neg_split]):
            dst_path = os.path.join(new_dataset_base_path, label)
            os.makedirs(dst_path)
            for file in sampled_files:
                shutil.copy2(file, dst_path)

        split_index += 1


def get_all_file_paths_per_label(dataset_path, label):
    files_paths = []
    label_path = os.path.join(dataset_path, label)
    for filename in tqdm(os.listdir(label_path), desc=f'Loading {label} reviews'):
        files_paths.append(os.path.join(label_path, filename))

    return files_paths


def get_imbdb_bert_base_uncased_datasets(data_path):
    base_path = os.path.join(data_path, f"bert-base-uncased-cached")

    if not os.path.isdir(base_path):
        texts, labels = load_imdb_data(data_path)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize and encode the dataset
        encodings, labels = _get_encodings_and_labels(tokenizer, texts, labels)

        # save data for future reuse
        os.makedirs(base_path)
        torch.save(encodings, os.path.join(base_path, ENCODINGS_PT))
        torch.save(labels, os.path.join(base_path, LABELS_PT))

    else:
        encodings = torch.load(os.path.join(base_path, ENCODINGS_PT))
        labels = torch.load(os.path.join(base_path, LABELS_PT))

    return CustomTensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)


def get_plain_imdb_data(data_path):
    base_path = os.path.join(data_path, f"bert-base-uncased-cached")

    return load_imdb_data(data_path)


def _get_encodings_and_labels(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    labels = torch.tensor(labels)
    return encodings, labels


def _reduced_data(indices, texts, labels):
    reduced_texts, reduced_labels = [], []
    for i in indices:
        reduced_texts.append(texts[i])
        reduced_labels.append(labels[i])
    return reduced_texts, reduced_labels


if __name__ == '__main__':
    root_data_path = '/tmp/pycharm_project_761/data'
    # create_sub_dataset("/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train", 100)
    # create_sub_dataset("/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test", 100)
    # create_sub_dataset("/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/train", 1000)
    # create_sub_dataset("/Users/nils/uni/programming/jit-LLM/data/imdb/data/aclImdb/test", 1000)
    train_path = os.path.join(root_data_path, "imdb/data/aclImdb/train")
    test_path = os.path.join(root_data_path, "imdb/data/aclImdb/test")
    create_dataset_splits(os.path.join(train_path), 250)
    create_dataset_splits(os.path.join(test_path), 250)

    print("test")
