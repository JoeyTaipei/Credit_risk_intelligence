"""Train LightGBM and CatBoost with the same 5-fold split as XGBoost v2."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.features_v2 import TARGET_COL, load_credit_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
N_SPLITS = 5
INNER_SPLITS = 3
EARLY_STOPPING_ROUNDS = 50
XGB_BEST_PARAMS_PATH = PROCESSED_DIR / "xgb_best_params.json"


def load_feature_target(data_path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load v2 tabular features and split them into X/y."""
    feature_df = load_credit_features_v2() if data_path is None else load_credit_features_v2(data_path)
    X = feature_df.drop(columns=[TARGET_COL])
    y = feature_df[TARGET_COL].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def confirm_xgb_data_path(data_path: Path | str) -> None:
    """Verify the Step 2 XGBoost params were produced from the same dataset."""
    if not XGB_BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost params file: {XGB_BEST_PARAMS_PATH}")

    with XGB_BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
        xgb_params = json.load(file)

    recorded_path = xgb_params.get("data_path")
    if recorded_path is None:
        raise ValueError(
            f"{XGB_BEST_PARAMS_PATH} has data_path=null; rerun Step 2 with --data-path {data_path}."
        )

    expected = Path(data_path).resolve()
    recorded = Path(recorded_path).resolve()
    if recorded != expected:
        raise ValueError(
            f"XGBoost data_path mismatch: expected {expected}, found {recorded}."
        )

    print(f"Confirmed XGBoost Step 2 data_path: {recorded_path}")


def positive_class_weight(y: pd.Series | np.ndarray) -> float:
    """Return n_negative / n_positive for imbalanced binary training."""
    y_array = np.asarray(y)
    n_pos = int(np.sum(y_array == 1))
    n_neg = int(np.sum(y_array == 0))
    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive labels.")
    return n_neg / n_pos


def suggest_lgbm_params(trial: optuna.Trial, scale_pos_weight: float) -> dict[str, Any]:
    """Suggest LightGBM hyperparameters using the requested search pattern."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "min_split_gain": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _fit_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any],
) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def run_lgbm_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Optimize LightGBM parameters against 3-fold CV AUC."""
    scale_pos_weight = positive_class_weight(y)
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial, scale_pos_weight)
        fold_aucs: list[float] = []
        for train_idx, val_idx in inner_cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = _fit_lgbm_model(X_train, y_train, X_val, y_val, params)
            val_prob = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_val, val_prob)))
        return float(np.mean(fold_aucs))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = suggest_lgbm_params(study.best_trial, scale_pos_weight)
    best_params["best_inner_cv_auc"] = float(study.best_value)
    best_params["n_optuna_trials"] = int(n_trials)
    return best_params


def train_lgbm_folds(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    """Train 5 LightGBM fold models and save OOF predictions."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    model_params = {
        key: value
        for key, value in params.items()
        if key not in {"best_inner_cv_auc", "n_optuna_trials", "data_path"}
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = _fit_lgbm_model(X_train, y_train, X_val, y_val, model_params)
        val_prob = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        fold_auc = float(roc_auc_score(y_val, val_prob))
        fold_aucs.append(fold_auc)
        print(f"LightGBM fold {fold} AUC: {fold_auc:.6f}")

        with (PROCESSED_DIR / f"lgbm_fold_{fold}.pkl").open("wb") as file:
            pickle.dump(model, file)

    np.save(PROCESSED_DIR / "lgbm_oof.npy", oof)
    return oof, fold_aucs


def train_catboost_folds(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[float]]:
    """Train 5 CatBoost fold models and save OOF predictions."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric="AUC",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_STATE,
            loss_function="Logloss",
            verbose=False,
            thread_count=-1,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        fold_auc = float(roc_auc_score(y_val, val_prob))
        fold_aucs.append(fold_auc)
        print(f"CatBoost fold {fold} AUC: {fold_auc:.6f}")

        with (PROCESSED_DIR / f"catboost_fold_{fold}.pkl").open("wb") as file:
            pickle.dump(model, file)

    np.save(PROCESSED_DIR / "catboost_oof.npy", oof)
    return oof, fold_aucs


def _print_cv_summary(model_name: str, y: pd.Series, oof: np.ndarray, fold_aucs: list[float]) -> None:
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    global_auc = float(roc_auc_score(y, oof))
    print(f"{model_name} OOF AUC mean +/- std: {mean_auc:.6f} +/- {std_auc:.6f}")
    print(f"{model_name} global OOF AUC: {global_auc:.6f}")


def run_training(
    n_lgbm_trials: int = 50,
    data_path: Path | str = Path("data/raw/cs-training2.csv"),
) -> dict[str, float]:
    """Run LightGBM Optuna search, LightGBM CV, and CatBoost CV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    confirm_xgb_data_path(data_path)
    X, y = load_feature_target(data_path)
    print(f"Loaded v2 features: rows={len(X):,}, columns={X.shape[1]}")

    lgbm_params = run_lgbm_optuna(X, y, n_trials=n_lgbm_trials)
    lgbm_params["data_path"] = str(data_path)
    with (PROCESSED_DIR / "lgbm_best_params.json").open("w", encoding="utf-8") as file:
        json.dump(lgbm_params, file, indent=2)

    lgbm_oof, lgbm_fold_aucs = train_lgbm_folds(X, y, lgbm_params)
    catboost_oof, catboost_fold_aucs = train_catboost_folds(X, y)

    _print_cv_summary("LightGBM", y, lgbm_oof, lgbm_fold_aucs)
    _print_cv_summary("CatBoost", y, catboost_oof, catboost_fold_aucs)

    return {
        "lgbm_oof_auc": float(roc_auc_score(y, lgbm_oof)),
        "catboost_oof_auc": float(roc_auc_score(y, catboost_oof)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lgbm-trials",
        type=int,
        default=50,
        help="Number of LightGBM Optuna trials. Default matches project spec: 50.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/cs-training2.csv"),
        help="CSV path for Step 3. Default: data/raw/cs-training2.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(n_lgbm_trials=args.lgbm_trials, data_path=args.data_path)


if __name__ == "__main__":
    main()
