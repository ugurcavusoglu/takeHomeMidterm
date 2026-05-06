import math
import random
from collections import Counter, defaultdict
from config import SEED, NGRAM_ORDER

random.seed(SEED)


def build_ngram_lm(tokens: list[str], n: int) -> tuple[dict, dict, int]:
    vocab = set(tokens)
    vocab_size = len(vocab)

    ngram_counts: dict = defaultdict(Counter)
    context_counts: Counter = Counter()

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i: i + n - 1])
        word = tokens[i + n - 1]
        ngram_counts[context][word] += 1
        context_counts[context] += 1

    return dict(ngram_counts), dict(context_counts), vocab_size


def ngram_perplexity(
    ngram_counts: dict,
    context_counts: dict,
    vocab_size: int,
    tokens: list[str],
    n: int,
) -> float:
    log_prob_sum = 0.0
    count = 0

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i: i + n - 1])
        word = tokens[i + n - 1]
        word_count = ngram_counts.get(context, {}).get(word, 0)
        ctx_count = context_counts.get(context, 0)
        # Laplace (add-1) smoothing
        prob = (word_count + 1) / (ctx_count + vocab_size)
        log_prob_sum += math.log(prob)
        count += 1

    avg_log_prob = log_prob_sum / count
    return math.exp(-avg_log_prob)


def generate_text(
    ngram_counts: dict,
    context_counts: dict,
    vocab_size: int,
    seed_tokens: list[str],
    n: int,
    num_words: int = 20,
) -> str:
    tokens = list(seed_tokens)

    for _ in range(num_words):
        context = tuple(tokens[-(n - 1):])
        candidates = ngram_counts.get(context, {})
        ctx_count = context_counts.get(context, 0)

        if not candidates:
            break

        words = list(candidates.keys())
        counts = [candidates[w] + 1 for w in words]
        total = sum(counts)
        probs = [c / total for c in counts]
        tokens.append(random.choices(words, weights=probs)[0])

    return " ".join(tokens)


if __name__ == "__main__":
    from evaluate import load_wikitext
    train_tokens, _, test_tokens, _ = load_wikitext()
    ngram_counts, context_counts, vocab_size = build_ngram_lm(train_tokens, NGRAM_ORDER)
    ppl = ngram_perplexity(ngram_counts, context_counts, vocab_size, test_tokens, NGRAM_ORDER)
    print(f"N-gram (n={NGRAM_ORDER}) Test Perplexity: {ppl:.2f}")
    sample = generate_text(ngram_counts, context_counts, vocab_size, ["the", "president"], NGRAM_ORDER)
    print(f"Sample: {sample}")
