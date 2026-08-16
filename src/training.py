"""
Model training, hyperparameter tuning, and evaluation.

Trains multiple classifiers with and without SMOTE resampling
via GridSearchCV with stratified cross-validation.
Selects the best model by Balanced Error Rate.
"""

import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
)
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    TEST_SIZE, CV_FOLDS, RANDOM_STATE, SCORING_METRIC,
    N_JOBS, PARAM_GRIDS, TARGET, SMOTE_K_NEIGHBORS,
)
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


# ─── Model Definitions ───────────────────────────────────

def _get_classifiers():
    """Return a dict of {name: classifier_instance}."""
    return {
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        ),
    }


def _build_pipelines(preprocessor, classifiers):
    """
    For each classifier, build TWO pipelines:
      1. Base: preprocessor → classifier
      2. SMOTE: preprocessor → SMOTE → classifier

    Returns dict of {display_name: (pipeline, param_grid)}
    """
    pipelines = {}

    for name, clf in classifiers.items():
        if name not in PARAM_GRIDS:
            continue

        # --- Base pipeline (no resampling) ---
        base_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])
        pipelines[name] = (base_pipe, PARAM_GRIDS[name])

        # --- SMOTE pipeline ---
        smote_pipe = ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=RANDOM_STATE,
            )),
            ("clf", clf.__class__(**clf.get_params())),  # fresh copy
        ])
        pipelines[f"SMOTE+{name}"] = (smote_pipe, PARAM_GRIDS[name])

    return pipelines


# ─── Train / Evaluate ────────────────────────────────────

def split_data(df):
    """Stratified train/validation split preserving class balance."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Split → train: %d  val: %d  (class dist: %s)",
        len(X_train), len(X_val), y_train.value_counts().to_dict(),
    )

    return X_train, X_val, y_train, y_val


def train_all_models(X_train, y_train, X_val, y_val):
    """
    Train every pipeline (base + SMOTE variants) with GridSearchCV
    and evaluate on the validation set.
    """
    preprocessor = build_preprocessor()
    classifiers = _get_classifiers()
    pipelines = _build_pipelines(preprocessor, classifiers)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for name, (pipeline, param_grid) in pipelines.items():
        print(f"\n{'═' * 55}")
        print(f"  Training: {name}")
        print(f"{'═' * 55}")

        t0 = time.time()

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=SCORING_METRIC,
            cv=cv,
            n_jobs=N_JOBS,
            verbose=1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        elapsed = time.time() - t0

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)

        # probability scores for ROC/PR curves
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_val)[:, 1]
        else:
            y_prob = best_model.decision_function(X_val)

        ba = balanced_accuracy_score(y_val, y_pred)
        ber = 1 - ba
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)

        results[name] = {
            "best_cv": grid.best_score_,
            "val_bal_acc": ba,
            "val_BER": ber,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "classification_report": classification_report(y_val, y_pred),
            "best_params": grid.best_params_,
            "model": best_model,
            "y_prob": y_prob,
            "train_time": elapsed,
        }

        print(f"  Best CV Balanced Accuracy : {grid.best_score_:.4f}")
        print(f"  Val  Balanced Accuracy    : {ba:.4f}")
        print(f"  Val  BER                  : {ber:.4f}")
        print(f"  Val  ROC-AUC              : {roc_auc:.4f}")
        print(f"  Val  PR-AUC               : {pr_auc:.4f}")
        print(f"  Best Params               : {grid.best_params_}")
        print(f"  Training Time             : {elapsed:.1f}s")

    return results


def select_best_model(results):
    """Return (name, result_dict) for the model with lowest validation BER."""
    best_name = min(results, key=lambda k: results[k]["val_BER"])
    r = results[best_name]
    print(f"\n{'─' * 55}")
    print(f"  ✓ Best model: {best_name}")
    print(f"    BER = {r['val_BER']:.4f}  |  ROC-AUC = {r['roc_auc']:.4f}  |  PR-AUC = {r['pr_auc']:.4f}")
    print(f"{'─' * 55}")
    return best_name, results[best_name]


def build_comparison_table(results):
    """Return a sorted DataFrame comparing all trained models."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "CV Bal Acc": round(r["best_cv"], 4),
            "Val Bal Acc": round(r["val_bal_acc"], 4),
            "Val BER": round(r["val_BER"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
            "PR-AUC": round(r["pr_auc"], 4),
            "Train Time (s)": round(r["train_time"], 1),
        })

    return pd.DataFrame(rows).sort_values("Val BER").reset_index(drop=True)
