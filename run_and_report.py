#!/usr/bin/env python3
"""
run_and_report.py — Run the full pipeline and print README-ready results.
"""

import warnings
warnings.filterwarnings("ignore")

import time
from src.data_loader import load_datasets, summarize
from src.training import (
    split_data, train_all_models, select_best_model, build_comparison_table,
)
from src.predict import retrain_and_predict, save_predictions

print("\n" + "=" * 60)
print("  IMBALANCED CLASSIFICATION PIPELINE — FULL RUN")
print("  (Base + SMOTE × 4 models = 8 experiments)")
print("=" * 60)

labeled, unlabeled, unlabeled_idx = load_datasets()
summarize(labeled, "Labeled")

X_train, X_val, y_train, y_val = split_data(labeled)

t0 = time.time()
results = train_all_models(X_train, y_train, X_val, y_val)
total_time = time.time() - t0

comparison = build_comparison_table(results)

print("\n")
print("=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(comparison.to_string(index=False))

best_name, best_result = select_best_model(results)
print(f"\n  Best params: {best_result['best_params']}")
print(f"\n  Classification Report:\n{best_result['classification_report']}")

# Predict
submission = retrain_and_predict(best_result["model"], labeled, unlabeled, unlabeled_idx)
out_path = save_predictions(submission)

# README table
print("\n")
print("=" * 60)
print("  COPY-PASTE THIS INTO YOUR README.md")
print("=" * 60)
print()
print("| Model | CV Balanced Acc | Val Balanced Acc | Val BER | ROC-AUC | PR-AUC | Status |")
print("|---|---|---|---|---|---|---|")

for idx, row in comparison.iterrows():
    name = row["Model"]
    cv = row["CV Bal Acc"]
    val = row["Val Bal Acc"]
    ber = row["Val BER"]
    roc = row["ROC-AUC"]
    pr = row["PR-AUC"]

    if name == best_name:
        print(f"| **{name}** | **{cv}** | **{val}** | **{ber}** | **{roc}** | **{pr}** | **Selected** |")
    else:
        print(f"| {name} | {cv} | {val} | {ber} | {roc} | {pr} | |")

print()
print(f"Total training time: {total_time/60:.1f} minutes")
print(f"Predictions saved to: {out_path}")
print()
print("  ✓ Done! Update your README.md with the table above.")
print("  ✓ Then run: python main.py --save-plots")
print("     to generate all charts (ROC, PR, SMOTE impact, etc.)")
print()
