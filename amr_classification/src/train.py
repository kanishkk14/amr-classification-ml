"""
train.py — Model training for the AMR classification pipeline.
"""

from sklearn.ensemble import RandomForestClassifier


def train_rf_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with class-balanced weighting
    to handle the typical class imbalance in AMR datasets.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
