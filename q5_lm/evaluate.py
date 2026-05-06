import json
import os
import math
import torch
from datasets import load_dataset
from config import SEED, DATASET_NAME, DATASET_VERSION, NGRAM_ORDER, RESULTS_DIR
from ngram_lm import build_ngram_lm, ngram_perplexity, generate_text as ngram_generate
from lstm_lm import build_vocab_and_loaders, train_lstm, lstm_perplexity, generate_text as lstm_generate


def load_wikitext() -> tuple[list[str], list[str], list[str], list[str]]:
    ds = load_dataset(DATASET_NAME, DATASET_VERSION)

    def tokenize(split: str) -> list[str]:
        tokens = []
        for line in ds[split]["text"]:
            line = line.strip()
            if line:
                tokens.extend(line.lower().split())
        return tokens

    train_tokens = tokenize("train")
    val_tokens = tokenize("validation")
    test_tokens = tokenize("test")

    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,} | Test tokens: {len(test_tokens):,}")
    return train_tokens, val_tokens, test_tokens, list(set(train_tokens))


def show_qualitative_examples(
    ngram_counts: dict,
    context_counts: dict,
    vocab_size: int,
    lstm_model,
    vocab,
    device: torch.device,
    n: int = NGRAM_ORDER,
) -> None:
    seeds = [["the", "president"], ["in", "the"], ["he", "was"]]
    print(f"\n{'='*80}")
    print("QUALITATIVE EXAMPLES (Generated Text)")
    print("=" * 80)
    for seed in seeds:
        ngram_text = ngram_generate(ngram_counts, context_counts, vocab_size, seed, n, num_words=15)
        lstm_text = lstm_generate(lstm_model, vocab, seed, num_words=15, device=device)
        print(f"\nSeed: '{' '.join(seed)}'")
        print(f"  N-gram : {ngram_text}")
        print(f"  LSTM   : {lstm_text}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tokens, val_tokens, test_tokens, _ = load_wikitext()

    ngram_cache = f"{RESULTS_DIR}/ngram_perplexity.json"
    if os.path.exists(ngram_cache):
        print("\nLoading cached N-gram perplexity...")
        with open(ngram_cache) as f:
            ngram_result = json.load(f)
        ngram_ppl = ngram_result["perplexity"]
        ngram_counts, context_counts, vocab_size = build_ngram_lm(train_tokens, NGRAM_ORDER)
    else:
        print(f"\nBuilding {NGRAM_ORDER}-gram LM...")
        ngram_counts, context_counts, vocab_size = build_ngram_lm(train_tokens, NGRAM_ORDER)
        ngram_ppl = ngram_perplexity(ngram_counts, context_counts, vocab_size, test_tokens, NGRAM_ORDER)
        print(f"N-gram Test Perplexity: {ngram_ppl:.2f}")
        with open(ngram_cache, "w") as f:
            json.dump({"n": NGRAM_ORDER, "perplexity": ngram_ppl}, f)

    vocab, train_loader, val_loader = build_vocab_and_loaders(train_tokens, val_tokens)

    lstm_cache = f"{RESULTS_DIR}/lstm_best.pt"
    if os.path.exists(lstm_cache):
        print("\nLoading cached LSTM model...")
        from lstm_lm import LSTMLangModel
        lstm_model = LSTMLangModel(len(vocab), 256, 512, 2, 0.5).to(device)
        lstm_model.load_state_dict(torch.load(lstm_cache, map_location=device))
    else:
        print("\nTraining LSTM LM...")
        lstm_model = train_lstm(train_loader, val_loader, vocab, device)

    from lstm_lm import LMDataset
    from torch.utils.data import DataLoader as DL
    from config import MAX_LEN, BATCH_SIZE
    test_ds = LMDataset(test_tokens, vocab, MAX_LEN)
    test_loader = DL(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    lstm_ppl = lstm_perplexity(lstm_model, test_loader, device)
    print(f"LSTM Test Perplexity: {lstm_ppl:.2f}")

    print(f"\n=== Comparison Table ===")
    print(f"{'Model':<30} {'Test Perplexity':>16}")
    print("-" * 48)
    print(f"{'N-gram (trigram)':<30} {ngram_ppl:>16.2f}")
    print(f"{'LSTM (2-layer)':<30} {lstm_ppl:>16.2f}")

    results = {
        "ngram": {"n": NGRAM_ORDER, "perplexity": ngram_ppl},
        "lstm": {"perplexity": lstm_ppl},
    }
    with open(f"{RESULTS_DIR}/q5_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    show_qualitative_examples(ngram_counts, context_counts, vocab_size, lstm_model, vocab, device)


if __name__ == "__main__":
    main()
