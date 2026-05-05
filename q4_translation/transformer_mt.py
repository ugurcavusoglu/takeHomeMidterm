import json
import torch
from transformers import pipeline
from tqdm import tqdm
from config import MARIAN_MODEL, MARIAN_BATCH_SIZE, MAX_LEN, RESULTS_DIR


def run_marian(sentences: list[str]) -> list[str]:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")

    translator = pipeline(
        "translation_en_to_de",
        model=MARIAN_MODEL,
        device=device,
        max_length=MAX_LEN,
    )

    translations = []
    for i in tqdm(range(0, len(sentences), MARIAN_BATCH_SIZE), desc="MarianMT inference"):
        batch = sentences[i: i + MARIAN_BATCH_SIZE]
        results = translator(batch, truncation=True)
        translations.extend([r["translation_text"] for r in results])

    with open(f"{RESULTS_DIR}/marian_translations.json", "w") as f:
        json.dump(translations, f)

    return translations


if __name__ == "__main__":
    from preprocess import load_multi30k
    _, _, _, _, _, test_src, _ = load_multi30k()
    translations = run_marian(test_src)
    print(f"\nGenerated {len(translations)} translations.")
    print(f"Example:\n  SRC: {test_src[0]}\n  TGT: {translations[0]}")
