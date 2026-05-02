import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from metrics import analyze_ner_errors
from config import (
    SEED, MAX_LEN, BERT_BATCH_SIZE, EPOCHS_BERT, LR_BERT,
    CHECKPOINT_DIR, RESULTS_DIR, DATASET_NAME,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "bert-base-cased"


def get_label_list(ds) -> list[str]:
    return ds["train"].features["ner_tags"].feature.names


def tokenize_and_align(batch: dict, tokenizer, label_list: list[str]) -> dict:
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
    )
    all_labels = []
    for i, ner_tags in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(ner_tags[word_id])
            else:
                labels.append(-100)
            prev_word_id = word_id
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized


def make_compute_metrics(label_list: list[str]):
    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)

        true_tags, pred_tags = [], []
        for pred_seq, label_seq in zip(preds, labels):
            true_row, pred_row = [], []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    true_row.append(label_list[l])
                    pred_row.append(label_list[p])
            true_tags.append(true_row)
            pred_tags.append(pred_row)

        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags),
        }
    return compute_metrics


def main() -> None:
    print("=== BERT NER (bert-base-cased) ===\n")

    ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    label_list = get_label_list(ds)
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_ds = ds.map(
        lambda b: tokenize_and_align(b, tokenizer, label_list),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{CHECKPOINT_DIR}/bert_ner",
        num_train_epochs=EPOCHS_BERT,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=LR_BERT,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("\nValidation:")
    val_result = trainer.predict(tokenized_ds["validation"])
    _print_ner_report(val_result, tokenized_ds["validation"]["labels"], label_list)

    print("\nTest:")
    test_result = trainer.predict(tokenized_ds["test"])
    test_true, test_pred = _print_ner_report(test_result, tokenized_ds["test"]["labels"], label_list)

    test_tokens = [ex["tokens"] for ex in ds["test"]]
    analyze_ner_errors(test_tokens, test_true, test_pred, n=10)


def _print_ner_report(result, raw_labels, label_list: list[str]) -> tuple:
    preds = np.argmax(result.predictions, axis=2)
    true_tags, pred_tags = [], []
    for pred_seq, label_seq in zip(preds, raw_labels):
        true_row, pred_row = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_row.append(label_list[l])
                pred_row.append(label_list[p])
        true_tags.append(true_row)
        pred_tags.append(pred_row)

    print(f"  Precision : {precision_score(true_tags, pred_tags):.4f}")
    print(f"  Recall    : {recall_score(true_tags, pred_tags):.4f}")
    print(f"  F1        : {f1_score(true_tags, pred_tags):.4f}")
    print(classification_report(true_tags, pred_tags))
    return true_tags, pred_tags


if __name__ == "__main__":
    main()
