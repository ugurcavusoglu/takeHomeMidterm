import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import (
    SEED, MAX_LEN, BATCH_SIZE, EPOCHS, LR, CLIP,
    EARLY_STOPPING_PATIENCE, LSTM_EMB_DIM, LSTM_HID_DIM, LSTM_LAYERS, DROPOUT,
    RESULTS_DIR,
)

random.seed(SEED)
torch.manual_seed(SEED)

PAD_IDX = 0


class Vocab:
    def __init__(self, tokens: list[str], min_freq: int = 2) -> None:
        from collections import Counter
        counter = Counter(tokens)
        words = ["<pad>", "<unk>"] + [w for w, c in counter.most_common() if c >= min_freq]
        self.token2idx: dict[str, int] = {w: i for i, w in enumerate(words)}
        self.idx2token: list[str] = words

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, 1)

    def decode(self, idx: int) -> str:
        return self.idx2token[idx] if idx < len(self.idx2token) else "<unk>"


class LMDataset(Dataset):
    def __init__(self, tokens: list[str], vocab: Vocab, seq_len: int) -> None:
        ids = [vocab.encode(t) for t in tokens]
        self.sequences = []
        for i in range(0, len(ids) - seq_len, seq_len):
            x = ids[i: i + seq_len]
            y = ids[i + 1: i + seq_len + 1]
            self.sequences.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]


class LSTMLangModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)
        return self.fc(self.dropout(output))


def build_vocab_and_loaders(
    train_tokens: list[str],
    val_tokens: list[str],
) -> tuple[Vocab, DataLoader, DataLoader]:
    vocab = Vocab(train_tokens)
    train_ds = LMDataset(train_tokens, vocab, MAX_LEN)
    val_ds = LMDataset(val_tokens, vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Vocab size: {len(vocab)} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return vocab, train_loader, val_loader


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: Vocab,
    device: torch.device,
) -> LSTMLangModel:
    model = LSTMLangModel(len(vocab), LSTM_EMB_DIM, LSTM_HID_DIM, LSTM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_loss += loss.item()

        val_loss = lstm_perplexity(model, val_loader, device, return_loss=True)
        avg_train = train_loss / len(train_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {math.exp(val_loss):.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{RESULTS_DIR}/lstm_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/lstm_best.pt", map_location=device))
    return model


def lstm_perplexity(
    model: LSTMLangModel,
    data_loader: DataLoader,
    device: torch.device,
    return_loss: bool = False,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.reshape(-1, output.shape[-1]), y.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss if return_loss else math.exp(avg_loss)


def generate_text(
    model: LSTMLangModel,
    vocab: Vocab,
    seed_words: list[str],
    num_words: int = 20,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    tokens = list(seed_words)
    with torch.no_grad():
        for _ in range(num_words):
            ids = torch.tensor([[vocab.encode(t) for t in tokens[-MAX_LEN:]]], dtype=torch.long).to(device)
            output = model(ids)
            probs = torch.softmax(output[0, -1], dim=0)
            next_id = torch.multinomial(probs, 1).item()
            tokens.append(vocab.decode(next_id))
    return " ".join(tokens)
