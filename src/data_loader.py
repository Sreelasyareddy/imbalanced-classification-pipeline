"""
Data loading, validation, and exploratory analysis.
"""

import pandas as pd
import numpy as np
import logging
from src.config import (
    LABELED_PATH, UNLABELED_PATH, INDEX_COL, TARGET,
    ORDINAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES,
)

logger = logging.getLogger(__name__)


def load_datasets(labeled_path=None, unlabeled_path=None):
    """
    Load labeled and unlabeled Excel datasets, strip the index column,
    and return clean DataFrames.

    Returns
    -------
    labeled : pd.DataFrame   (features + target)
    unlabeled : pd.DataFrame  (features only)
    unlabeled_index : pd.Series  (original row indices for submission)
    """
    labeled_path = labeled_path or LABELED_PATH
    unlabeled_path = unlabeled_path or UNLABELED_PATH

    logger.info("Loading labeled data from %s", labeled_path)
    labeled = pd.read_excel(labeled_path)

    logger.info("Loading unlabeled data from %s", unlabeled_path)
    unlabeled = pd.read_excel(unlabeled_path)

    # preserve original indices for submission file
    unlabeled_index = unlabeled[INDEX_COL].copy()

    # drop the Excel row-index column
    labeled = labeled.drop(columns=[INDEX_COL])
    unlabeled = unlabeled.drop(columns=[INDEX_COL])

    logger.info(
        "Labeled: %d samples × %d features  |  Unlabeled: %d samples",
        *labeled.shape, len(unlabeled),
    )

    return labeled, unlabeled, unlabeled_index


def summarize(df, name="dataset"):
    """Print a concise summary of a DataFrame for quick inspection."""
    print(f"\n{'─' * 50}")
    print(f"  {name.upper()} SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")

    if TARGET in df.columns:
        dist = df[TARGET].value_counts()
        ratio = dist.min() / dist.max()
        print(f"  Class balance  : {dist.to_dict()}  (minority ratio: {ratio:.2f})")

    missing = df.isnull().sum()
    cols_with_na = missing[missing > 0]
    if len(cols_with_na):
        print(f"  Missing values : {len(cols_with_na)} columns have NaNs")
        for col, n in cols_with_na.items():
            print(f"    {col:>6s}: {n:,}  ({n / len(df) * 100:.1f}%)")
    else:
        print("  Missing values : none")

    print(f"  Feature groups :")
    print(f"    Ordinal  ({len(ORDINAL_FEATURES)}): {ORDINAL_FEATURES}")
    print(f"    Numeric  ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"    Binary  ({len(BINARY_FEATURES)}): {BINARY_FEATURES}")
    print()
