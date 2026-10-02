"""
synthetic_example.py — End-to-end demo of the AMR classification pipeline
                        using synthetic genomic feature data.

Synthetic data note
-------------------
Features gyrA_S83L and parC_S80I are causal (used in label generation).
Features gyrA_D87N and qnrS1 are noise (not used in label generation).
Permutation importance should recover this signal vs. noise distinction.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate_schema   import validate_input_data
from src.split_by_mlst     import get_grouped_splits
from src.train             import train_rf_model
from src.evaluate          import (
    tune_classification_threshold,
    calculate_metrics,
    get_permutation_importance,
)

SCRIPT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_mock_data(n_samples: int = 400) -> pd.DataFrame:
    """
    Generate synthetic AMR dataset with realistic genomic features.
    gyrA_S83L and parC_S80I are causal; gyrA_D87N and qnrS1 are noise.
    """
    np.random.seed(42)

    mlst_groups = ["ST" + str(np.random.randint(1, 2000)) for _ in range(40)]
    groups      = np.random.choice(mlst_groups, size=n_samples)

    data = {
        "Genome_ID":       [f"ERR{1000 + i}" for i in range(n_samples)],
        "MLST_Group":      groups,
        "gyrA_S83L":       np.random.binomial(1, 0.20, n_samples),  # causal
        "gyrA_D87N":       np.random.binomial(1, 0.15, n_samples),  # noise
        "parC_S80I":       np.random.binomial(1, 0.10, n_samples),  # causal
        "qnrS1":           np.random.binomial(1, 0.05, n_samples),  # noise
    }

    base_risk  = 0.10 + (data["gyrA_S83L"] * 0.35) + (data["parC_S80I"] * 0.25)
    probs      = np.clip(base_risk + np.random.normal(0, 0.15, n_samples), 0, 1)
    data["Phenotype_Resistant"] = np.random.binomial(1, probs)

    return pd.DataFrame(data)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run() -> None:
    print("=== AMR Classification Pipeline (Synthetic Demo) ===\n")

    # 1. Generate and validate data
    df = generate_mock_data()
    validate_input_data(df)
    print(f"Dataset: {len(df)} genomes | "
          f"Resistant: {df['Phenotype_Resistant'].sum()} "
          f"({df['Phenotype_Resistant'].mean()*100:.1f}%)")

    feature_cols = [c for c in df.columns
                    if c not in ("Genome_ID", "MLST_Group", "Phenotype_Resistant")]
    X      = df[feature_cols].reset_index(drop=True)
    y      = df["Phenotype_Resistant"].reset_index(drop=True)
    groups = df["MLST_Group"].reset_index(drop=True)

    # 2. Outer CV loop (MLST-grouped, 5 folds)
    outer_splits = get_grouped_splits(X, y, groups, n_splits=5)
    fold_metrics = []
    importances  = []

    print(f"\nRunning {len(outer_splits)}-fold MLST-grouped nested CV...\n")

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]
        groups_train     = groups.iloc[train_idx]

        # Inner CV for threshold tuning (3 folds)
        inner_splits   = get_grouped_splits(X_train, y_train, groups_train, n_splits=3)
        pooled_y_val   = []
        pooled_y_proba = []

        for inner_train_idx, inner_val_idx in inner_splits:
            X_in_train = X_train.iloc[inner_train_idx]
            y_in_train = y_train.iloc[inner_train_idx]
            X_in_val   = X_train.iloc[inner_val_idx]
            y_in_val   = y_train.iloc[inner_val_idx]

            model_inner    = train_rf_model(X_in_train, y_in_train)
            y_in_val_proba = model_inner.predict_proba(X_in_val)[:, 1]

            pooled_y_val.extend(y_in_val.tolist())
            pooled_y_proba.extend(y_in_val_proba.tolist())

        # Threshold selected on pooled inner-fold validation set
        optimal_threshold = tune_classification_threshold(
            np.array(pooled_y_val), np.array(pooled_y_proba)
        )

        # Outer model trained on full outer-train set
        model_outer    = train_rf_model(X_train, y_train)
        y_test_proba   = model_outer.predict_proba(X_test)[:, 1]

        metrics           = calculate_metrics(y_test, y_test_proba, optimal_threshold)
        metrics["Fold"]   = fold + 1
        fold_metrics.append(metrics)

        # Permutation importance on held-out test set
        imp = get_permutation_importance(model_outer, X_test, y_test)
        importances.append(imp)

        print(f"  Fold {fold+1}  AUROC={metrics['AUROC']:.3f}  "
              f"PR-AUC={metrics['PR-AUC']:.3f}  "
              f"F1={metrics['F1']:.3f}  "
              f"Threshold={metrics['Threshold']:.3f}")

    # 3. Aggregate results
    metrics_df    = pd.DataFrame(fold_metrics)
    metric_cols   = ["AUROC", "PR-AUC", "F1", "Precision", "Recall"]
    summary_stats = metrics_df[metric_cols].agg(["mean", "std"]).T
    summary_stats.columns = ["Mean", "Std"]
    summary_stats.index.name = "Metric"

    # Fix #4: importances arrays may differ in length across folds —
    # average only the feature-axis (axis=0 on a stacked array is safe
    # only if all arrays have the same length = n_features, which they do
    # since features are fixed; test set size varies but imp length = n_features)
    avg_importances = np.mean(np.vstack(importances), axis=0)
    importance_df   = (
        pd.DataFrame({"Feature": feature_cols, "Importance": avg_importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Summary Metrics (mean ± std across folds) ===")
    for metric, row in summary_stats.iterrows():
        print(f"  {metric:<10s}: {row['Mean']:.3f} ± {row['Std']:.3f}")

    # 4. Save CSVs
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)
    summary_stats.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"))
    importance_df.to_csv(os.path.join(RESULTS_DIR, "permutation_importances.csv"), index=False)
    print(f"\nCSVs saved to {RESULTS_DIR}/")

    # 5. Plots
    _plot_cv_metrics(metrics_df, metric_cols)
    _plot_permutation_importance(importance_df)
    _plot_pr_roc_summary(summary_stats)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_cv_metrics(metrics_df: pd.DataFrame, metric_cols: list) -> None:
    """Box + strip plot of per-fold metric distributions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data    = [metrics_df[m].values for m in metric_cols]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    colors = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#B71C1C"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold points
    for i, col in enumerate(metric_cols):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(metrics_df))
        ax.scatter(np.ones(len(metrics_df)) * (i + 1) + jitter,
                   metrics_df[col], color="white", edgecolor="black",
                   zorder=3, s=40, linewidths=0.8)

    ax.set_xticks(range(1, len(metric_cols) + 1))
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Validation Performance — MLST-Grouped Nested CV\n"
                 "(5 outer folds, Random Forest)", fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Random baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cv_metrics_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def _plot_permutation_importance(importance_df: pd.DataFrame) -> None:
    """Horizontal bar chart of mean permutation importances."""
    df = importance_df.sort_values("Importance", ascending=True)
    colors = ["#E53935" if imp > 0.01 else "#90A4AE" for imp in df["Importance"]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.6)))
    ax.barh(df["Feature"], df["Importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean Permutation Importance (Δ PR-AUC)")
    ax.set_title("Feature Importance — Held-Out Permutation Analysis\n"
                 "(averaged across 5 outer folds × 10 repeats)",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "permutation_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def _plot_pr_roc_summary(summary_stats: pd.DataFrame) -> None:
    """Bar chart of mean ± std for each metric."""
    metrics = summary_stats.index.tolist()
    means   = summary_stats["Mean"].values
    stds    = summary_stats["Std"].values
    colors  = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#B71C1C"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="white",
                  error_kw={"linewidth": 1.5, "ecolor": "black"})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Summary Performance — Mean ± Std across Folds",
                 fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "summary_metrics_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


if __name__ == "__main__":
    run()
