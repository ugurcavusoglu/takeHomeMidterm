import json
import torch
from transformers import pipeline
from tqdm import tqdm
from config import BART_MODEL, MAX_SUMMARY_LEN, MIN_SUMMARY_LEN, BART_BATCH_SIZE, RESULTS_DIR


def run_bart(articles: list[str]) -> list[str]:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")

    summarizer = pipeline(
        "summarization",
        model=BART_MODEL,
        device=device,
        truncation=True,
    )

    summaries = []
    for i in tqdm(range(0, len(articles), BART_BATCH_SIZE), desc="BART inference"):
        batch = articles[i: i + BART_BATCH_SIZE]
        results = summarizer(
            batch,
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            truncation=True,
        )
        summaries.extend([r["summary_text"] for r in results])

    with open(f"{RESULTS_DIR}/bart_summaries.json", "w") as f:
        json.dump(summaries, f)

    return summaries


if __name__ == "__main__":
    from preprocess import load_cnn_dailymail
    articles, _ = load_cnn_dailymail()
    summaries = run_bart(articles)
    print(f"\nGenerated {len(summaries)} BART summaries.")
    print(f"Example:\n{summaries[0]}")
