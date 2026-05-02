import time
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import load_imdb
from evaluate import compute_metrics, analyze_misclassifications
from config import SEED, RESULTS_DIR

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    print("=== TF-IDF + Logistic Regression ===\n")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_imdb()

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    t0 = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, train_labels)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s\n")

    print("Validation:")
    val_preds = clf.predict(X_val)
    compute_metrics(val_labels, val_preds)

    print("\nTest:")
    test_preds = clf.predict(X_test)
    compute_metrics(test_labels, test_preds)

    np.save(f"{RESULTS_DIR}/tfidf_preds.npy", test_preds)
    np.save(f"{RESULTS_DIR}/test_labels.npy", test_labels)

    analyze_misclassifications(test_texts, test_labels, test_preds.tolist(), n=10)


if __name__ == "__main__":
    main()
