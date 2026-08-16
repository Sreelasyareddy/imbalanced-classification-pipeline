"""
Feature preprocessing pipeline using scikit-learn ColumnTransformer.

Handles three feature types in a single leak-proof pipeline:
  - Ordinal  → mode-impute → OrdinalEncoder
  - Numeric  → mean-impute → StandardScaler
  - Binary   → mode-impute (passthrough)
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import ORDINAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES


def build_preprocessor():
    """
    Construct a ColumnTransformer that applies the correct
    imputation and encoding strategy to each feature group.

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )),
                ]),
                ORDINAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]),
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
