import random
import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate
from evaluate import compute_metrics as local_compute_metrics, analyze_misclassifications
from config import (
    SEED, MAX_LEN, BATCH_SIZE, EPOCHS_BERT, LR_BERT,
    CHECKPOINT_DIR, RESULTS_DIR, DATASET_NAME,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
accuracy_metric = hf_evaluate.load("accuracy")
f1_metric = hf_evaluate.load("f1")


def tokenize(batch: dict, tokenizer: AutoTokenizer) -> dict:
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")


def hf_compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}


def main() -> None:
    print("=== DistilBERT Fine-tuning ===\n")

    ds = hf_load_dataset(DATASET_NAME)
    train_full = ds["train"].shuffle(seed=SEED)
    val_size = int(0.1 * len(train_full))
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_tok = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_tok = val_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    test_tok = test_ds.map(lambda b: tokenize(b, tokenizer), batched=True)

    for split in [train_tok, val_tok, test_tok]:
        split.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"{CHECKPOINT_DIR}/bert",
        num_train_epochs=EPOCHS_BERT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR_BERT,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("\nValidation:")
    val_result = trainer.predict(val_tok)
    val_preds = np.argmax(val_result.predictions, axis=1).tolist()
    val_labels = val_tok["label"].tolist()
    local_compute_metrics(val_labels, val_preds)

    print("\nTest:")
    test_result = trainer.predict(test_tok)
    test_preds = np.argmax(test_result.predictions, axis=1).tolist()
    test_labels = test_tok["label"].tolist()
    local_compute_metrics(test_labels, test_preds)

    np.save(f"{RESULTS_DIR}/bert_preds.npy", np.array(test_preds))

    test_texts = [d["text"] for d in test_ds]
    analyze_misclassifications(test_texts, test_labels, test_preds, n=10)


if __name__ == "__main__":
    main()
