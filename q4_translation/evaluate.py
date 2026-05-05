import json
import os
import torch
import nltk
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn
from tqdm import tqdm
from preprocess import load_multi30k
from seq2seq import train_seq2seq, generate_translations
from transformer_mt import run_marian
from config import RESULTS_DIR

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    return bleu.corpus_score(predictions, [references]).score / 100.0


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        scores.append(meteor_score([ref_tokens], pred_tokens))
    return float(np.mean(scores))


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    chrf = CHRF()
    return chrf.corpus_score(predictions, [references]).score / 100.0


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bert_score_fn(predictions, references, lang="de", verbose=False)
    return float(f1.mean())


def compute_all_metrics(predictions: list[str], references: list[str], model_name: str) -> dict:
    print(f"\n=== {model_name} ===")
    bleu = compute_bleu(predictions, references)
    meteor = compute_meteor(predictions, references)
    chrf = compute_chrf(predictions, references)
    bertscore = compute_bertscore(predictions, references)

    print(f"  BLEU      : {bleu:.4f}")
    print(f"  METEOR    : {meteor:.4f}")
    print(f"  ChrF      : {chrf:.4f}")
    print(f"  BERTScore : {bertscore:.4f}")

    return {"BLEU": bleu, "METEOR": meteor, "ChrF": chrf, "BERTScore": bertscore}


def show_qualitative_examples(
    src_sentences: list[str],
    references: list[str],
    seq2seq_preds: list[str],
    marian_preds: list[str],
    n: int = 3,
) -> None:
    print(f"\n{'='*80}")
    print("QUALITATIVE EXAMPLES")
    print("=" * 80)
    for i in range(n):
        print(f"\n--- Example {i+1} ---")
        print(f"SOURCE    : {src_sentences[i]}")
        print(f"REFERENCE : {references[i]}")
        print(f"SEQ2SEQ   : {seq2seq_preds[i]}")
        print(f"MARIAN    : {marian_preds[i]}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, test_src, test_tgt = load_multi30k()

    seq2seq_cache = f"{RESULTS_DIR}/seq2seq_translations.json"
    if os.path.exists(seq2seq_cache):
        print("\nLoading cached Seq2Seq translations...")
        with open(seq2seq_cache) as f:
            seq2seq_preds = json.load(f)
    else:
        print("\nTraining Seq2Seq...")
        model = train_seq2seq(train_loader, val_loader, src_vocab, tgt_vocab, device)
        seq2seq_preds = generate_translations(model, test_src, src_vocab, tgt_vocab, device)
        with open(seq2seq_cache, "w") as f:
            json.dump(seq2seq_preds, f)

    marian_cache = f"{RESULTS_DIR}/marian_translations.json"
    if os.path.exists(marian_cache):
        print("\nLoading cached MarianMT translations...")
        with open(marian_cache) as f:
            marian_preds = json.load(f)
    else:
        print("\nRunning MarianMT...")
        marian_preds = run_marian(test_src)

    seq2seq_metrics = compute_all_metrics(seq2seq_preds, test_tgt, "Seq2Seq + Bahdanau Attention")
    marian_metrics = compute_all_metrics(marian_preds, test_tgt, "MarianMT (opus-mt-en-de)")

    print("\n=== Comparison Table ===")
    print(f"{'Metric':<12} {'Seq2Seq':>10} {'MarianMT':>10}")
    print("-" * 34)
    for key in ["BLEU", "METEOR", "ChrF", "BERTScore"]:
        print(f"{key:<12} {seq2seq_metrics[key]:>10.4f} {marian_metrics[key]:>10.4f}")

    results = {"seq2seq": seq2seq_metrics, "marian": marian_metrics}
    with open(f"{RESULTS_DIR}/q4_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    show_qualitative_examples(test_src, test_tgt, seq2seq_preds, marian_preds, n=3)


if __name__ == "__main__":
    main()
