import numpy as np
from datasets import load_dataset
from config import SEED, DATASET_NAME, DATASET_VERSION, NUM_SAMPLES


def load_cnn_dailymail(num_samples: int = NUM_SAMPLES) -> tuple[list[str], list[str]]:
    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split="test")
    ds = ds.shuffle(seed=SEED).select(range(num_samples))

    articles = ds["article"]
    highlights = ds["highlights"]

    avg_article_len = np.mean([len(a.split()) for a in articles])
    avg_summary_len = np.mean([len(h.split()) for h in highlights])

    print(f"Loaded {len(articles)} samples from CNN/DailyMail test split")
    print(f"Avg article length : {avg_article_len:.0f} words")
    print(f"Avg summary length : {avg_summary_len:.0f} words")

    return list(articles), list(highlights)
