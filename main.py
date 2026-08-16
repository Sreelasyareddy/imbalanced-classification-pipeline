#!/usr/bin/env python3
"""
main.py — End-to-end binary classification pipeline for imbalanced data.

Usage
-----
    python main.py                    # train + predict (default)
    python main.py --train-only       # train and evaluate, skip prediction
    python main.py --save-plots       # save all figures to output/

Run `python main.py -h` for full options.
"""

import argparse
import logging
import warnings

from src.config import OUTPUT_DIR, ORDINAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES
from src.data_loader import load_datasets, summarize
from src.training import (
    split_data, train_all_models, select_best_model, build_comparison_table,
)
from src.predict import retrain_and_predict, save_predictions
from src.visualize import (
    plot_class_distribution, plot_correlation_matrix,
    plot_model_comparison, plot_confusion_matrices,
    plot_roc_curves, plot_precision_recall_curves,
    plot_feature_importance, plot_smote_impact,
)

warnings.filterwarnings("ignore")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Binary classification pipeline for imbalanced tabular data.",
    )
    p.add_argument("--train-only", action="store_true",
                    help="Train and evaluate without generating predictions.")
    p.add_argument("--save-plots", action="store_true",
                    help="Save all figures as PNG to output/.")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Enable debug logging.")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("pipeline")

    # ── 1. Load Data ──────────────────────────────────────
    logger.info("Loading datasets…")
    labeled, unlabeled, unlabeled_index = load_datasets()
    summarize(labeled, "labeled")

    # ── 2. EDA Plots ──────────────────────────────────────
    fig_dist = plot_class_distribution(labeled)
    fig_corr = plot_correlation_matrix(labeled)

    # ── 3. Split ──────────────────────────────────────────
    X_train, X_val, y_train, y_val = split_data(labeled)

    # ── 4. Train All Models (Base + SMOTE) ────────────────
    results = train_all_models(X_train, y_train, X_val, y_val)

    # ── 5. Compare ────────────────────────────────────────
    comparison = build_comparison_table(results)
    print("\n" + comparison.to_string(index=False))

    best_name, best_result = select_best_model(results)
    print(f"\n  Classification Report ({best_name}):")
    print(best_result["classification_report"])

    # ── 6. Evaluation Plots ───────────────────────────────
    fig_comp = plot_model_comparison(comparison)
    fig_cm = plot_confusion_matrices(results)
    fig_roc = plot_roc_curves(results, y_val)
    fig_pr = plot_precision_recall_curves(results, y_val)
    fig_smote = plot_smote_impact(comparison)

    # Feature importance (if tree-based model won)
    all_features = ORDINAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
    fig_fi = plot_feature_importance(best_result["model"], all_features)

    # ── 7. Predict ────────────────────────────────────────
    if not args.train_only:
        submission = retrain_and_predict(
            best_result["model"], labeled, unlabeled, unlabeled_index,
        )
        out_path = save_predictions(submission)
        print(f"\n  Predictions saved → {out_path}")

    # ── 8. Save Plots ────────────────────────────────────
    if args.save_plots:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figures = [
            ("class_distribution", fig_dist),
            ("correlation_matrix", fig_corr),
            ("model_comparison", fig_comp),
            ("confusion_matrices", fig_cm),
            ("roc_curves", fig_roc),
            ("precision_recall_curves", fig_pr),
            ("smote_impact", fig_smote),
            ("feature_importance", fig_fi),
        ]
        for name, fig in figures:
            if fig is not None:
                path = OUTPUT_DIR / f"{name}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                logger.info("Saved %s", path)

    print("\n  ✓ Pipeline complete.\n")


if __name__ == "__main__":
    main()
