import numpy as np
from collections import Counter
from datasets import load_dataset
from config import SEED, DATASET_NAME, MAX_VOCAB, GLOVE_PATH, GLOVE_DIM, MAX_LEN

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "O"


def load_conll() -> tuple:
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)

    def extract(split):
        tokens = [example["tokens"] for example in split]
        tags = [
            [split.features["ner_tags"].feature.int2str(t) for t in example["ner_tags"]]
            for example in split
        ]
        return tokens, tags

    train_tokens, train_tags = extract(ds["train"])
    val_tokens, val_tags = extract(ds["validation"])
    test_tokens, test_tags = extract(ds["test"])

    print(f"Train: {len(train_tokens)} | Val: {len(val_tokens)} | Test: {len(test_tokens)} sentences")
    return train_tokens, train_tags, val_tokens, val_tags, test_tokens, test_tags


def build_word_vocab(token_lists: list[list[str]], max_vocab: int = MAX_VOCAB) -> dict[str, int]:
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(t.lower() for t in tokens)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def build_tag_vocab(tag_lists: list[list[str]]) -> dict[str, int]:
    # PAD maps to 0, used for masking in CRF and loss
    all_tags = sorted({tag for tags in tag_lists for tag in tags})
    vocab = {PAD_TOKEN: 0}
    for tag in all_tags:
        if tag not in vocab:
            vocab[tag] = len(vocab)
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


def encode_sequences(
    token_lists: list[list[str]],
    tag_lists: list[list[str]],
    word_vocab: dict[str, int],
    tag_vocab: dict[str, int],
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    unk_idx = word_vocab[UNK_TOKEN]
    pad_word = word_vocab[PAD_TOKEN]
    pad_tag = tag_vocab[PAD_TOKEN]

    X, y, lengths = [], [], []
    for tokens, tags in zip(token_lists, tag_lists):
        toks = tokens[:max_len]
        tgs = tags[:max_len]
        lengths.append(len(toks))

        word_ids = [word_vocab.get(t.lower(), unk_idx) for t in toks]
        tag_ids = [tag_vocab[t] for t in tgs]

        word_ids += [pad_word] * (max_len - len(word_ids))
        tag_ids += [pad_tag] * (max_len - len(tag_ids))

        X.append(word_ids)
        y.append(tag_ids)

    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64), lengths
