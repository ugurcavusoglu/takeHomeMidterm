import json
import os
import nltk
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn
from tqdm import tqdm
from preprocess import load_cnn_dailymail
from textrank import run_textrank
from train_bart import run_bart
from config import RESULTS_DIR

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "ROUGE-1": np.mean(r1),
        "ROUGE-2": np.mean(r2),
        "ROUGE-L": np.mean(rl),
    }


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return result.score / 100.0


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        scores.append(meteor_score([ref_tokens], pred_tokens))
    return float(np.mean(scores))


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def compute_all_metrics(predictions: list[str], references: list[str], model_name: str) -> dict:
    print(f"\n=== {model_name} ===")
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    meteor = compute_meteor(predictions, references)
    bertscore = compute_bertscore(predictions, references)

    print(f"  ROUGE-1   : {rouge['ROUGE-1']:.4f}")
    print(f"  ROUGE-2   : {rouge['ROUGE-2']:.4f}")
    print(f"  ROUGE-L   : {rouge['ROUGE-L']:.4f}")
    print(f"  BLEU      : {bleu:.4f}")
    print(f"  METEOR    : {meteor:.4f}")
    print(f"  BERTScore : {bertscore:.4f}")

    return {**rouge, "BLEU": bleu, "METEOR": meteor, "BERTScore": bertscore}


def show_qualitative_examples(
    articles: list[str],
    references: list[str],
    textrank_preds: list[str],
    bart_preds: list[str],
    n: int = 3,
) -> None:
    print(f"\n{'='*80}")
    print("QUALITATIVE EXAMPLES")
    print('='*80)
    for i in range(n):
        print(f"\n--- Example {i+1} ---")
        print(f"SOURCE (first 300 chars):\n{articles[i][:300]}...")
        print(f"\nREFERENCE:\n{references[i]}")
        print(f"\nTEXTRANK:\n{textrank_preds[i]}")
        print(f"\nBART:\n{bart_preds[i]}")
        print()


def main() -> None:
    articles, references = load_cnn_dailymail()

    print("\nRunning TextRank...")
    textrank_preds = run_textrank(articles)

    bart_cache = f"{RESULTS_DIR}/bart_summaries.json"
    if os.path.exists(bart_cache):
        print("\nLoading cached BART summaries...")
        with open(bart_cache) as f:
            bart_preds = json.load(f)
    else:
        print("\nRunning BART...")
        bart_preds = run_bart(articles)

    textrank_metrics = compute_all_metrics(textrank_preds, references, "TextRank (Extractive)")
    bart_metrics = compute_all_metrics(bart_preds, references, "BART (Abstractive)")

    print("\n=== Comparison Table ===")
    print(f"{'Metric':<12} {'TextRank':>10} {'BART':>10}")
    print("-" * 34)
    for key in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR", "BERTScore"]:
        print(f"{key:<12} {textrank_metrics[key]:>10.4f} {bart_metrics[key]:>10.4f}")

    results = {"textrank": textrank_metrics, "bart": bart_metrics}
    with open(f"{RESULTS_DIR}/q3_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    show_qualitative_examples(articles, references, textrank_preds, bart_preds, n=3)


if __name__ == "__main__":
    main()
