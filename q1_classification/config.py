import os

SEED = 42
DATASET_NAME = "imdb"

MAX_LEN = 256
BATCH_SIZE = 32
MAX_VOCAB = 30000
GLOVE_DIM = 100

EPOCHS_BILSTM = 5
EPOCHS_BERT = 3
LR_BILSTM = 1e-3
LR_BERT = 2e-5
PATIENCE = 2

GLOVE_PATH = "glove.6B.100d.txt"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
