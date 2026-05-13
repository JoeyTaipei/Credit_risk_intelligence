"""Optuna-tuned 5-fold XGBoost training on v2 tabular features."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data.features_v2 import TARGET_COL, load_credit_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
N_SPLITS = 5
INNER_SPLITS = 3
EARLY_STOPPING_ROUNDS = 50


def load_feature_target(data_path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load v2 tabular features and split them into X/y."""
    feature_df = load_credit_features_v2() if data_path is None else load_credit_features_v2(data_path)
    X = feature_df.drop(columns=[TARGET_COL])
    y = feature_df[TARGET_COL].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def positive_class_weight(y: pd.Series | np.ndarray) -> float:
    """Return n_negative / n_positive for imbalanced binary training."""
    y_array = np.asarray(y)
    n_pos = int(np.sum(y_array == 1))
    n_neg = int(np.sum(y_array == 0))
    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive labels.")
    return n_neg / n_pos


def suggest_xgb_params(trial: optuna.Trial, scale_pos_weight: float) -> dict[str, Any]:
    """Suggest XGBoost hyperparameters from the requested search space."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def _fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any],
) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 100,
) -> dict[str, Any]:
    """Optimize XGBoost parameters against 3-fold CV AUC."""
    scale_pos_weight = positive_class_weight(y)
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, scale_pos_weight)
        fold_aucs: list[float] = []

        for train_idx, val_idx in inner_cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = _fit_xgb_model(X_train, y_train, X_val, y_val, params)
            val_prob = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_val, val_prob)))

        return float(np.mean(fold_aucs))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = suggest_xgb_params(study.best_trial, scale_pos_weight)
    best_params["best_inner_cv_auc"] = float(study.best_value)
    best_params["n_optuna_trials"] = int(n_trials)
    return best_params


def train_folds(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    """Train 5 fold models, save artifacts, and return OOF predictions/AUCs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    shap_values_oof = np.zeros((len(y), X.shape[1]), dtype=np.float32)

    model_params = {
        key: value
        for key, value in params.items()
        if key not in {"best_inner_cv_auc", "n_optuna_trials", "data_path"}
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = _fit_xgb_model(X_train, y_train, X_val, y_val, model_params)
        val_prob = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        fold_auc = float(roc_auc_score(y_val, val_prob))
        fold_aucs.append(fold_auc)
        print(f"XGBoost fold {fold} AUC: {fold_auc:.6f}")

        with (PROCESSED_DIR / f"xgb_fold_{fold}.pkl").open("wb") as file:
            pickle.dump(model, file)

        explainer = shap.TreeExplainer(model)
        shap_result = explainer(X_val)
        shap_values = getattr(shap_result, "values", shap_result)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        shap_values_oof[val_idx] = shap_values.astype(np.float32)

    np.save(PROCESSED_DIR / "xgb_oof.npy", oof)
    np.save(PROCESSED_DIR / "xgb_shap_values.npy", shap_values_oof)
    with (PROCESSED_DIR / "xgb_shap_feature_names.json").open("w", encoding="utf-8") as file:
        json.dump(list(X.columns), file, indent=2)

    return oof, fold_aucs


def run_training(
    n_trials: int = 100,
    data_path: Path | str | None = None,
) -> dict[str, float]:
    """Run Optuna search followed by tuned 5-fold XGBoost training."""
    X, y = load_feature_target(data_path)
    print(f"Loaded v2 features: rows={len(X):,}, columns={X.shape[1]}")
    print(f"Positive rate: {float(y.mean()):.4f}")

    best_params = run_optuna_search(X, y, n_trials=n_trials)
    best_params["data_path"] = str(data_path) if data_path is not None else None
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "xgb_best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    oof, fold_aucs = train_folds(X, y, best_params)
    oof_auc = float(roc_auc_score(y, oof))
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    print(f"XGBoost OOF AUC mean +/- std: {mean_auc:.6f} +/- {std_auc:.6f}")
    print(f"XGBoost global OOF AUC: {oof_auc:.6f}")
    return {
        "oof_auc": oof_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials. Default matches project spec: 100.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to data/raw/cs-training.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(n_trials=args.n_trials, data_path=args.data_path)


if __name__ == "__main__":
    main()
