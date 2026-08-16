"""
Central configuration for the classification pipeline.
All hyperparameters, paths, and feature definitions live here
so nothing is hard-coded across modules.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

LABELED_PATH = DATA_DIR / "labeled.xlsx"
UNLABELED_PATH = DATA_DIR / "unlabeled.xlsx"

# ─── Feature Groups ──────────────────────────────────────
ORDINAL_FEATURES = ["x2", "x3", "x4"]
NUMERIC_FEATURES = ["x15", "x16", "x17", "x18", "x19", "x20", "x21"]
BINARY_FEATURES = [
    "x1", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
]
TARGET = "label"
INDEX_COL = "Unnamed: 0"

# ─── Training ────────────────────────────────────────────
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
SCORING_METRIC = "balanced_accuracy"
N_JOBS = -1  # use all cores

# ─── Resampling Strategies ───────────────────────────────
# Each model will be benchmarked with and without SMOTE
SMOTE_K_NEIGHBORS = 5

# ─── Hyperparameter Grids ────────────────────────────────
# Each grid is tried TWICE: once raw, once with SMOTE prefix
PARAM_GRIDS = {
    "RandomForest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 15],
        "clf__min_samples_split": [2, 5],
        "clf__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.05, 0.1],
        "clf__scale_pos_weight": [1, 3],  # ~ratio of neg/pos
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    },
    "SVM_RBF": {
        "clf__C": [0.5, 1.0, 5.0],
        "clf__gamma": ["scale", "auto"],
        "clf__class_weight": ["balanced"],
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__class_weight": ["balanced"],
    },
}
