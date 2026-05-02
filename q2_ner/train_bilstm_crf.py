import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchcrf import CRF  # pip install pytorch-crf
from tqdm import tqdm
from preprocess import load_conll, build_word_vocab, build_tag_vocab, load_glove, encode_sequences
from metrics import compute_ner_metrics, ids_to_tags, analyze_ner_errors
from config import (
    SEED, MAX_LEN, BATCH_SIZE, EPOCHS_BILSTM, LR_BILSTM,
    PATIENCE, GRAD_CLIP, CHECKPOINT_DIR, RESULTS_DIR, GLOVE_DIM, MAX_VOCAB,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_tags: int, embedding_matrix: np.ndarray, pad_tag_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.pad_tag_idx = pad_tag_idx

    def forward(self, x: torch.Tensor, tags: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        emissions = self.fc(self.dropout(lstm_out))

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        return self.crf.decode(emissions, mask=mask)


def make_loader(X: np.ndarray, y: np.ndarray, lengths: list[int], shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.long)
    mask = torch.zeros(len(X), X.shape[1], dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return DataLoader(TensorDataset(X_t, y_t, mask), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch, mask_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = model(X_batch, tags=y_batch, mask=mask_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> list[list[int]]:
    model.eval()
    all_preds = []
    for X_batch, _, mask_batch in loader:
        X_batch = X_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        preds = model(X_batch, mask=mask_batch)
        all_preds.extend(preds)
    return all_preds


def main() -> None:
    print("=== BiLSTM-CRF NER ===\n")
    train_tok, train_tags, val_tok, val_tags, test_tok, test_tags = load_conll()

    word_vocab = build_word_vocab(train_tok, max_vocab=MAX_VOCAB)
    tag_vocab = build_tag_vocab(train_tags)
    id2tag = {v: k for k, v in tag_vocab.items()}
    num_tags = len(tag_vocab)
    pad_tag_idx = tag_vocab["<PAD>"]

    embedding_matrix = load_glove(word_vocab, dim=GLOVE_DIM)

    X_train, y_train, train_len = encode_sequences(train_tok, train_tags, word_vocab, tag_vocab, MAX_LEN)
    X_val, y_val, val_len = encode_sequences(val_tok, val_tags, word_vocab, tag_vocab, MAX_LEN)
    X_test, y_test, test_len = encode_sequences(test_tok, test_tags, word_vocab, tag_vocab, MAX_LEN)

    train_loader = make_loader(X_train, y_train, train_len, shuffle=True)
    val_loader = make_loader(X_val, y_val, val_len, shuffle=False)
    test_loader = make_loader(X_test, y_test, test_len, shuffle=False)

    model = BiLSTMCRF(len(word_vocab), GLOVE_DIM, hidden_dim=256,
                      num_tags=num_tags, embedding_matrix=embedding_matrix,
                      pad_tag_idx=pad_tag_idx).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BILSTM)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS_BILSTM + 1):
        train_loss = train_epoch(model, train_loader, optimizer)

        val_preds_ids = predict(model, val_loader)
        val_true_tags = ids_to_tags([[t for t in row] for row in y_val.tolist()], val_len, id2tag)
        val_pred_tags = ids_to_tags(val_preds_ids, val_len, id2tag)

        from seqeval.metrics import f1_score
        val_f1 = f1_score(val_true_tags, val_pred_tags)
        print(f"Epoch {epoch}/{EPOCHS_BILSTM} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/bilstm_crf_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/bilstm_crf_best.pt", map_location=DEVICE))

    print("\nValidation:")
    val_preds_ids = predict(model, val_loader)
    val_true_tags = ids_to_tags([[t for t in row] for row in y_val.tolist()], val_len, id2tag)
    val_pred_tags = ids_to_tags(val_preds_ids, val_len, id2tag)
    compute_ner_metrics(val_true_tags, val_pred_tags)

    print("\nTest:")
    test_preds_ids = predict(model, test_loader)
    test_true_tags = ids_to_tags([[t for t in row] for row in y_test.tolist()], test_len, id2tag)
    test_pred_tags = ids_to_tags(test_preds_ids, test_len, id2tag)
    compute_ner_metrics(test_true_tags, test_pred_tags)

    analyze_ner_errors(test_tok, test_true_tags, test_pred_tags, n=10)


if __name__ == "__main__":
    main()
