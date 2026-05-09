"""XGBoost baseline model training utilities."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 20,
    "random_state": 42,
}


def _compute_metrics(y_true: Any, y_prob: np.ndarray) -> dict[str, float]:
    """Compute validation or test metrics from predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_xgb_baseline(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    use_smote: bool = True,
    random_state: int = 42,
) -> tuple:
    """Train XGBoost baseline. Return (model, val_metrics_dict).

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_val: Validation feature matrix.
        y_val: Validation target vector.
        use_smote: Whether to apply SMOTE to the training split only.
        random_state: Random seed for reproducible model fitting and SMOTE.

    Returns:
        Tuple of fitted XGBoost model and validation metrics dictionary.
    """
    X_fit = X_train
    y_fit = y_train

    if use_smote:
        from imblearn.over_sampling import SMOTE

        # SMOTE is applied only to training data to avoid leaking validation structure.
        X_fit, y_fit = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    params = {**DEFAULT_XGB_PARAMS, "random_state": random_state}
    model = XGBClassifier(**params)

    with mlflow.start_run():
        mlflow.log_params({**params, "use_smote": use_smote})
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        val_prob = predict(model, X_val)
        val_metrics = _compute_metrics(y_val, val_prob)
        # MLflow metric keys cannot contain "@"; keep returned keys per spec, log safe names.
        mlflow.log_metrics({name.replace("@", "_at_"): value for name, value in val_metrics.items()})
        mlflow.xgboost.log_model(model, "model")

    return model, val_metrics


def predict(model: Any, X: Any) -> np.ndarray:
    """Return positive-class probabilities for a fitted model.

    Args:
        model: Fitted model with a predict_proba method.
        X: Feature matrix.

    Returns:
        Positive-class probability array.
    """
    return model.predict_proba(X)[:, 1]
