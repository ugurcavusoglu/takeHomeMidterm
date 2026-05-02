import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_imdb, build_vocab, load_glove, texts_to_sequences
from metrics import compute_metrics, analyze_misclassifications
from config import (
    SEED, MAX_LEN, BATCH_SIZE, EPOCHS_BILSTM, LR_BILSTM,
    PATIENCE, CHECKPOINT_DIR, RESULTS_DIR, GLOVE_DIM, MAX_VOCAB,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, embedding_matrix: np.ndarray):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # concat forward and backward final hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out)).squeeze(1)


def make_loader(sequences: np.ndarray, labels: list, shuffle: bool) -> DataLoader:
    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[list, float]:
    model.eval()
    all_preds, total_loss = [], 0.0
    criterion = nn.BCEWithLogitsLoss()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
        all_preds.extend(preds)
    return all_preds, total_loss / len(loader)


def main() -> None:
    print("=== BiLSTM + GloVe ===\n")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_imdb()

    vocab = build_vocab(train_texts, max_vocab=MAX_VOCAB)
    embedding_matrix = load_glove(vocab, dim=GLOVE_DIM)

    train_seq = texts_to_sequences(train_texts, vocab, MAX_LEN)
    val_seq = texts_to_sequences(val_texts, vocab, MAX_LEN)
    test_seq = texts_to_sequences(test_texts, vocab, MAX_LEN)

    train_loader = make_loader(train_seq, train_labels, shuffle=True)
    val_loader = make_loader(val_seq, val_labels, shuffle=False)
    test_loader = make_loader(test_seq, test_labels, shuffle=False)

    model = BiLSTMClassifier(len(vocab), GLOVE_DIM, hidden_dim=128, embedding_matrix=embedding_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BILSTM)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS_BILSTM + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_preds, val_loss = predict(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS_BILSTM} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/bilstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/bilstm_best.pt", map_location=DEVICE))

    print("\nValidation:")
    val_preds, _ = predict(model, val_loader)
    compute_metrics(val_labels, val_preds)

    print("\nTest:")
    test_preds, _ = predict(model, test_loader)
    compute_metrics(test_labels, test_preds)

    np.save(f"{RESULTS_DIR}/bilstm_preds.npy", np.array(test_preds))

    analyze_misclassifications(test_texts, test_labels, test_preds, n=10)


if __name__ == "__main__":
    main()
