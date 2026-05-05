import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from config import BART_MODEL, MAX_INPUT_LEN, MAX_SUMMARY_LEN, MIN_SUMMARY_LEN, BART_BATCH_SIZE, RESULTS_DIR


def run_bart(articles: list[str]) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained(BART_MODEL)
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(device)
    model.eval()

    summaries = []
    for i in tqdm(range(0, len(articles), BART_BATCH_SIZE), desc="BART inference"):
        batch = articles[i: i + BART_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=MAX_SUMMARY_LEN,
                min_length=MIN_SUMMARY_LEN,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(decoded)

    with open(f"{RESULTS_DIR}/bart_summaries.json", "w") as f:
        json.dump(summaries, f)

    return summaries


if __name__ == "__main__":
    from preprocess import load_cnn_dailymail
    articles, _ = load_cnn_dailymail()
    summaries = run_bart(articles)
    print(f"\nGenerated {len(summaries)} BART summaries.")
    print(f"Example:\n{summaries[0]}")
