# CENG 467 – Natural Language Understanding and Generation
## Take-Home Midterm Examination

**Student:** Uğur Mert Çavuşoğlu  
**Student ID:** 300201087  
**Instructor:** Prof. Dr. Aytuğ Onan  
**Date:** May 2026

---

## Overview

This repository contains the code and report for the CENG 467 Take-Home Midterm. Five NLP tasks are implemented and compared across classical, neural, and transformer-based approaches.

All experiments were run on **Google Colab (T4 GPU)**. Random seed **42** is used throughout. Datasets are downloaded automatically via HuggingFace `datasets` at runtime — no manual data setup is needed.

---

## Repository Structure

```
├── report/
│   └── main.tex                  # LaTeX source for the full report
├── q1_classification/
│   ├── config.py                 # Hyperparameters and constants
│   ├── preprocess.py             # Tokenization and dataset loading
│   ├── train_tfidf.py            # TF-IDF + Logistic Regression
│   ├── train_bilstm.py           # BiLSTM + GloVe embeddings
│   ├── train_bert.py             # DistilBERT fine-tuning
│   ├── metrics.py                # Accuracy and Macro-F1 evaluation
│   └── q1_notebook.ipynb         # Colab notebook
├── q2_ner/
│   ├── config.py
│   ├── preprocess.py             # BIO tagging, token-label alignment
│   ├── train_bilstm_crf.py       # BiLSTM-CRF + GloVe
│   ├── train_bert_ner.py         # BERT token classification
│   └── metrics.py                # Per-entity P/R/F1
├── q3_summarization/
│   ├── config.py
│   ├── preprocess.py
│   ├── textrank.py               # Extractive summarization
│   ├── train_bart.py             # BART abstractive inference
│   └── evaluate.py               # ROUGE/BLEU/METEOR/BERTScore
├── q4_translation/
│   ├── config.py
│   ├── preprocess.py             # Multi30k loading, Vocab, DataLoader
│   ├── seq2seq.py                # Encoder-Decoder + Bahdanau Attention
│   ├── transformer_mt.py         # MarianMT (Helsinki-NLP/opus-mt-en-de)
│   └── evaluate.py               # BLEU/METEOR/ChrF/BERTScore
├── q5_lm/
│   ├── config.py
│   ├── ngram_lm.py               # Trigram LM with Laplace smoothing
│   ├── lstm_lm.py                # 2-layer LSTM language model
│   └── evaluate.py               # Perplexity + text generation
├── requirements.txt
├── CLAUDE.md
└── .gitignore
```

---

## Questions and Methods

### Q1 – Text Classification (IMDb)
| Model | Test Accuracy | Test Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 89.85% | 0.8985 |
| BiLSTM + GloVe | 85.28% | 0.8526 |
| DistilBERT (fine-tuned) | **91.15%** | **0.9115** |

### Q2 – Named Entity Recognition (CoNLL-2003)
| Model | Test Precision | Test Recall | Test F1 |
|---|---|---|---|
| BiLSTM-CRF + GloVe | 0.8639 | 0.7734 | 0.8161 |
| BERT (bert-base-cased) | 0.9050 | 0.9187 | **0.9118** |

### Q3 – Text Summarization (CNN/DailyMail, 1000 samples)
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore |
|---|---|---|---|---|---|---|
| TextRank (Extractive) | 0.3550 | 0.1415 | 0.2317 | 0.0921 | 0.3338 | 0.8637 |
| BART (Abstractive) | **0.4408** | **0.2127** | **0.3103** | **0.1700** | **0.3911** | **0.8803** |

### Q4 – Machine Translation (Multi30k EN→DE)
| Model | BLEU | METEOR | ChrF | BERTScore |
|---|---|---|---|---|
| Seq2Seq + Bahdanau Attention | 1.54 | 0.3827 | 0.3453 | 0.6687 |
| MarianMT (opus-mt-en-de) | **36.25** | **0.6668** | **0.6428** | **0.8967** |

### Q5 – Language Modeling (WikiText-2)
| Model | Test Perplexity |
|---|---|
| N-gram Trigram (Laplace) | 32,372.30 |
| LSTM 2-layer | **194.33** |

---

## Setup and Reproducibility

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run each question
```bash
# Q1 – Text Classification
cd q1_classification
python train_tfidf.py
python train_bilstm.py
python train_bert.py

# Q2 – NER
cd q2_ner
python train_bilstm_crf.py
python train_bert_ner.py

# Q3 – Summarization
cd q3_summarization
python train_bart.py
python evaluate.py

# Q4 – Machine Translation
cd q4_translation
python seq2seq.py
python evaluate.py

# Q5 – Language Modeling
cd q5_lm
python evaluate.py
```

> All scripts cache results to `results/` after the first run. Re-running loads from cache, skipping expensive inference.

---

## Environment

- Python 3.10+
- PyTorch 2.3.0
- HuggingFace Transformers 4.40.0
- HuggingFace Datasets 2.19.0
- Google Colab T4 GPU
- Random seed: 42

See `requirements.txt` for full dependency list.

---

## Notes

- Datasets are downloaded automatically at runtime via HuggingFace `datasets`. No manual download needed.
- Model checkpoints (`.pt` files) are saved to `results/` and excluded from git via `.gitignore`.
- BART and MarianMT are used inference-only (no fine-tuning). Seq2Seq and LSTM are trained from scratch.
- Test set is used **only once** for final evaluation in each question.
