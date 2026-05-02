import re
import random
import numpy as np
from collections import Counter
from datasets import load_dataset as hf_load_dataset
from config import SEED, DATASET_NAME, MAX_VOCAB, GLOVE_PATH, GLOVE_DIM

random.seed(SEED)
np.random.seed(SEED)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb() -> tuple:
    """Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)."""
    ds = hf_load_dataset(DATASET_NAME)

    train_full = ds["train"].shuffle(seed=SEED)
    val_size = int(0.1 * len(train_full))

    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = ds["test"]

    def extract(split):
        texts = [clean_text(t) for t in split["text"]]
        labels = list(split["label"])
        return texts, labels

    train_texts, train_labels = extract(train_ds)
    val_texts, val_labels = extract(val_ds)
    test_texts, test_labels = extract(test_ds)

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def build_vocab(texts: list[str], max_vocab: int = MAX_VOCAB) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def load_glove(vocab: dict[str, int], glove_path: str = GLOVE_PATH, dim: int = GLOVE_DIM) -> np.ndarray:
    embedding_matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embedding_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"GloVe: {found}/{len(vocab)} tokens found")
    return embedding_matrix


def texts_to_sequences(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    unk_idx = vocab[UNK_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    sequences = []
    for text in texts:
        tokens = text.split()[:max_len]
        ids = [vocab.get(t, unk_idx) for t in tokens]
        ids += [pad_idx] * (max_len - len(ids))
        sequences.append(ids)
    return np.array(sequences, dtype=np.int64)
