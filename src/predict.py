"""
Inference module — retrain best model on all labeled data,
then predict on the unlabeled set and save to CSV.
"""

import logging
import pandas as pd

from src.config import OUTPUT_DIR, TARGET

logger = logging.getLogger(__name__)


def retrain_and_predict(best_model, labeled_df, unlabeled_df, unlabeled_index):
    """
    Retrain the best pipeline on the *full* labeled dataset
    (train + val combined), then generate predictions on the
    unlabeled set.

    Parameters
    ----------
    best_model : sklearn.pipeline.Pipeline
    labeled_df : pd.DataFrame  (features + label)
    unlabeled_df : pd.DataFrame  (features only)
    unlabeled_index : pd.Series  (original row indices)

    Returns
    -------
    submission : pd.DataFrame  with columns [index, label]
    """
    X_full = labeled_df.drop(columns=[TARGET])
    y_full = labeled_df[TARGET]

    logger.info("Retraining on full labeled set (%d samples)…", len(X_full))
    best_model.fit(X_full, y_full)

    predictions = best_model.predict(unlabeled_df)

    dist = pd.Series(predictions).value_counts().to_dict()
    logger.info("Prediction distribution: %s", dist)

    submission = pd.DataFrame({
        "index": unlabeled_index,
        "label": predictions,
    })

    return submission


def save_predictions(submission, filename="predictions.csv"):
    """Write the submission DataFrame to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename

    submission.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)

    return out_path
