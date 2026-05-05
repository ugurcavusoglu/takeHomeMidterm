import random
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset
from config import SEED, DATASET_NAME, SRC_LANG, TGT_LANG, MAX_LEN, BATCH_SIZE

random.seed(SEED)
torch.manual_seed(SEED)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.token2idx: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.idx2token: list[str] = tokens

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2token[i] for i in indices if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]


def _tokenize(text: str) -> list[str]:
    return text.lower().strip().split()


def build_vocab(sentences: list[str], min_freq: int = 2) -> Vocab:
    counter: Counter = Counter()
    for sent in sentences:
        counter.update(_tokenize(sent))
    tokens = SPECIAL_TOKENS + [tok for tok, freq in counter.most_common() if freq >= min_freq]
    return Vocab(tokens)


def _encode_pair(
    src: str,
    tgt: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_ids = [SOS_IDX] + src_vocab.encode(_tokenize(src)[:MAX_LEN - 2]) + [EOS_IDX]
    tgt_ids = [SOS_IDX] + tgt_vocab.encode(_tokenize(tgt)[:MAX_LEN - 2]) + [EOS_IDX]
    return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def _make_collate(src_vocab: Vocab, tgt_vocab: Vocab):
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src, tgt in batch:
            s, t = _encode_pair(src, tgt, src_vocab, tgt_vocab)
            src_batch.append(s)
            tgt_batch.append(t)
        src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
        tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
        return src_padded, tgt_padded
    return collate_fn


def load_multi30k() -> tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab, list[str], list[str]]:
    ds = load_dataset(DATASET_NAME)
    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    src_vocab = build_vocab(train_data[SRC_LANG])
    tgt_vocab = build_vocab(train_data[TGT_LANG])

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Src vocab size: {len(src_vocab)} | Tgt vocab size: {len(tgt_vocab)}")

    collate_fn = _make_collate(src_vocab, tgt_vocab)

    train_loader = DataLoader(
        list(zip(train_data[SRC_LANG], train_data[TGT_LANG])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        list(zip(val_data[SRC_LANG], val_data[TGT_LANG])),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        list(zip(test_data[SRC_LANG], test_data[TGT_LANG])),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_src = list(test_data[SRC_LANG])
    test_tgt = list(test_data[TGT_LANG])

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, test_src, test_tgt
