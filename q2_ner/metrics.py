from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def compute_ner_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(classification_report(y_true, y_pred))
    return {"precision": p, "recall": r, "f1": f1}


def ids_to_tags(
    tag_ids: list[list[int]],
    lengths: list[int],
    id2tag: dict[int, str],
    pad_tag: str = "O",
) -> list[list[str]]:
    result = []
    for ids, length in zip(tag_ids, lengths):
        result.append([id2tag.get(i, pad_tag) for i in ids[:length]])
    return result


def analyze_ner_errors(
    token_lists: list[list[str]],
    y_true: list[list[str]],
    y_pred: list[list[str]],
    n: int = 10,
) -> None:
    errors = [
        (tokens, true, pred)
        for tokens, true, pred in zip(token_lists, y_true, y_pred)
        if true != pred
    ]
    print(f"\n--- NER Error Analysis ({min(n, len(errors))} of {len(errors)} sentences with errors) ---")
    for tokens, true, pred in errors[:n]:
        print(f"\n  Tokens : {' '.join(tokens)}")
        print(f"  True   : {' '.join(true)}")
        print(f"  Pred   : {' '.join(pred)}")
