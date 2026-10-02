"""
split_by_mlst.py — MLST-grouped cross-validation splitter.

Groups entire MLST sequence types into either train or test folds,
preventing lineage leakage where closely related genomes in both
sets would inflate performance estimates.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_grouped_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (train_indices, test_indices) tuples using
    GroupKFold so that no MLST group appears in both train and test.

    Indices are positional (iloc-compatible) and always contiguous
    regardless of the input DataFrame's index.
    """
    # Reset to positional index to guarantee iloc compatibility
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    g_reset = groups.reset_index(drop=True)

    gkf    = GroupKFold(n_splits=n_splits)
    splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in gkf.split(X_reset, y_reset, g_reset)
    ]
    return splits
