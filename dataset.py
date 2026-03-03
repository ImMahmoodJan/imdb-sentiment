"""
dataset.py — IMDB data loading, cleaning, tokenisation, and splitting.
"""

import re
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_vocab=20000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=256):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (max_len - len(ids))
    return ids


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def load_imdb(seed=42, val_frac=0.1):
    from datasets import load_dataset
    raw = load_dataset("imdb")
    all_train_texts  = [clean_text(x["text"]) for x in raw["train"]]
    all_train_labels = [x["label"] for x in raw["train"]]
    test_texts       = [clean_text(x["text"]) for x in raw["test"]]
    test_labels      = [x["label"] for x in raw["test"]]
    rng = random.Random(seed)
    indices = list(range(len(all_train_texts)))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_frac)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    train_texts  = [all_train_texts[i] for i in train_idx]
    train_labels = [all_train_labels[i] for i in train_idx]
    val_texts    = [all_train_texts[i] for i in val_idx]
    val_labels   = [all_train_labels[i] for i in val_idx]
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels