"""
Plotting utilities for exploratory analysis and model evaluation.

All figures are returned (not shown) so callers can choose
to display or save them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.config import NUMERIC_FEATURES, TARGET


def plot_class_distribution(df):
    """Bar chart of target class counts with imbalance ratio annotation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].value_counts()
    counts.plot(
        kind="bar", color=["#3b82f6", "#ef4444"], edgecolor="white", ax=ax,
    )
    ratio = counts.min() / counts.max()
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.annotate(
        f"Imbalance ratio: {ratio:.2f}",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", alpha=0.8),
    )
    fig.tight_layout()
    return fig


def plot_correlation_matrix(df):
    """Heatmap of Pearson correlations for numeric features."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[NUMERIC_FEATURES].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Matrix — Numerical Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_df):
    """Grouped bar chart comparing CV and validation balanced accuracy."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(comparison_df))
    w = 0.25

    ax.bar(x - w, comparison_df["CV Bal Acc"], w,
           label="CV Balanced Acc", color="#3b82f6", alpha=0.85)
    ax.bar(x, comparison_df["Val Bal Acc"], w,
           label="Val Balanced Acc", color="#10b981", alpha=0.85)
    ax.bar(x + w, comparison_df["ROC-AUC"], w,
           label="ROC-AUC", color="#8b5cf6", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.4, 0.85)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(results, top_n=4):
    """Confusion matrices for the top N models by BER."""
    sorted_models = sorted(results.items(), key=lambda x: x[1]["val_BER"])[:top_n]
    n = len(sorted_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, sorted_models):
        sns.heatmap(
            r["confusion_matrix"], annot=True, fmt="d",
            cmap="Blues", cbar=False, ax=ax,
            xticklabels=["0", "1"], yticklabels=["0", "1"],
        )
        ax.set_title(f"{name}\nBER = {r['val_BER']:.4f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Top Models",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_roc_curves(results, y_val):
    """Overlay ROC curves for all models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_models = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_models)))

    for (name, r), color in zip(sorted_models, colors):
        fpr, tpr, _ = roc_curve(y_val, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_precision_recall_curves(results, y_val):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_models = sorted(results.items(), key=lambda x: x[1]["pr_auc"], reverse=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_models)))

    for (name, r), color in zip(sorted_models, colors):
        precision, recall, _ = precision_recall_curve(y_val, r["y_prob"])
        ax.plot(recall, precision, label=f"{name} (AP={r['pr_auc']:.3f})",
                color=color, linewidth=2)

    # baseline = prevalence of positive class
    prevalence = y_val.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.4,
               label=f"Random (AP={prevalence:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Horizontal bar chart of feature importances (tree-based models only).
    """
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return None

    importances = clf.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        np.array(feature_names)[idx],
        importances[idx],
        color="#6366f1", edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_smote_impact(comparison_df):
    """
    Bar chart comparing base vs SMOTE variants side by side.
    """
    base = comparison_df[~comparison_df["Model"].str.startswith("SMOTE")]
    smote = comparison_df[comparison_df["Model"].str.startswith("SMOTE")]

    if smote.empty:
        return None

    # match base names to smote names
    base_names = base["Model"].tolist()
    pairs = []
    for bn in base_names:
        smote_name = f"SMOTE+{bn}"
        if smote_name in smote["Model"].values:
            b_ber = base[base["Model"] == bn]["Val BER"].values[0]
            s_ber = smote[smote["Model"] == smote_name]["Val BER"].values[0]
            pairs.append((bn, b_ber, s_ber))

    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [p[0] for p in pairs]
    base_bers = [p[1] for p in pairs]
    smote_bers = [p[2] for p in pairs]

    x = np.arange(len(names))
    w = 0.3

    bars1 = ax.bar(x - w / 2, base_bers, w, label="Base", color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x + w / 2, smote_bers, w, label="SMOTE", color="#ef4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Balanced Error Rate (lower = better)")
    ax.set_title("Impact of SMOTE on Model Performance",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # annotate improvement
    for i, (bn, b, s) in enumerate(pairs):
        diff = b - s
        symbol = "↓" if diff > 0 else "↑"
        color = "#10b981" if diff > 0 else "#ef4444"
        ax.annotate(
            f"{symbol} {abs(diff):.3f}",
            xy=(i, min(b, s) - 0.01), ha="center", fontsize=9,
            color=color, fontweight="bold",
        )

    fig.tight_layout()
    return fig
