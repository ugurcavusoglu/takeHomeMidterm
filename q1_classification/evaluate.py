import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(y_true: list, y_pred: list) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    return {"accuracy": acc, "macro_f1": f1}


def analyze_misclassifications(
    texts: list[str],
    y_true: list,
    y_pred: list,
    n: int = 10,
) -> None:
    label_names = {0: "Negative", 1: "Positive"}
    errors = [
        (texts[i], y_true[i], y_pred[i])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    print(f"\n--- Misclassification Analysis ({min(n, len(errors))} of {len(errors)} errors) ---")
    for text, true, pred in errors[:n]:
        snippet = text[:200].replace("\n", " ")
        print(f"\n  True: {label_names[true]} | Predicted: {label_names[pred]}")
        print(f"  Text: {snippet}...")
