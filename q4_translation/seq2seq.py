import random
import torch
import torch.nn as nn
from tqdm import tqdm
from config import (
    SEED, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT,
    EPOCHS, LR, CLIP, TEACHER_FORCING_RATIO, EARLY_STOPPING_PATIENCE,
    MAX_LEN, RESULTS_DIR,
)
from preprocess import PAD_IDX, SOS_IDX, EOS_IDX, Vocab

random.seed(SEED)
torch.manual_seed(SEED)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # Concatenate forward and backward final hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim)
        self.W2 = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        # hidden: [batch, hid_dim], encoder_outputs: [src_len, batch, hid_dim*2]
        src_len = encoder_outputs.shape[0]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs_t = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.W1(hidden_expanded) + self.W2(encoder_outputs_t))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim * 2, hid_dim, batch_first=False)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_token = tgt_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(tgt_token))
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        encoder_outputs_t = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights, encoder_outputs_t).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc_out(
            torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1)
        )
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = TEACHER_FORCING_RATIO,
    ) -> torch.Tensor:
        tgt_len, batch_size = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        dec_input = tgt[0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[t] = output
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[t] if use_teacher else output.argmax(1)

        return outputs

    def translate(self, src: torch.Tensor, max_len: int = MAX_LEN) -> list[int]:
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src)
            dec_input = torch.tensor([SOS_IDX], device=self.device)
            tokens = []
            for _ in range(max_len):
                output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
                pred = output.argmax(1)
                if pred.item() == EOS_IDX:
                    break
                tokens.append(pred.item())
                dec_input = pred
        return tokens


def build_model(src_vocab_size: int, tgt_vocab_size: int, device: torch.device) -> Seq2Seq:
    encoder = Encoder(src_vocab_size, ENC_EMB_DIM, HID_DIM, DROPOUT)
    decoder = Decoder(tgt_vocab_size, DEC_EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


def train_seq2seq(
    train_loader,
    val_loader,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
) -> Seq2Seq:
    model = build_model(len(src_vocab), len(tgt_vocab), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output_flat = output[1:].reshape(-1, len(tgt_vocab))
            tgt_flat = tgt[1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                output_flat = output[1:].reshape(-1, len(tgt_vocab))
                tgt_flat = tgt[1:].reshape(-1)
                val_loss += criterion(output_flat, tgt_flat).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{RESULTS_DIR}/seq2seq_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/seq2seq_best.pt", map_location=device))
    return model


def generate_translations(model: Seq2Seq, test_src: list[str], src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device) -> list[str]:
    model.eval()
    translations = []
    for sentence in tqdm(test_src, desc="Seq2Seq inference"):
        tokens = ["<sos>"] + sentence.lower().strip().split()[:MAX_LEN - 2] + ["<eos>"]
        src_ids = torch.tensor(src_vocab.encode(tokens[1:-1]), dtype=torch.long).unsqueeze(1).to(device)
        pred_ids = model.translate(src_ids)
        pred_tokens = tgt_vocab.decode(pred_ids)
        translations.append(" ".join(pred_tokens))
    return translations
