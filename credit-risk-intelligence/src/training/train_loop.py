"""Training entry points for Credit Risk Intelligence."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import clean_tabular, engineer_tabular_features
from src.models.xgb_baseline import predict, train_xgb_baseline
from src.utils.shap_visualizer import plot_shap_summary, plot_shap_waterfall


TARGET_COL = "SeriousDlqin2yrs"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CREDIT_PATH = PROJECT_ROOT / "data/raw/cs-training.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "docs/figures"


def _metrics(y_true: Any, y_prob: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics at the fixed 0.5 threshold."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 train, validation, and test splits."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[TARGET_COL],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[TARGET_COL],
    )
    return train_df, val_df, test_df


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist split DataFrames as parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)


def _feature_target(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target."""
    return split_df.drop(columns=[TARGET_COL]), split_df[TARGET_COL]


def _top_shap_features(shap_values: Any, feature_names: list[str]) -> list[str]:
    """Return the top three features by mean absolute SHAP value."""
    values = getattr(shap_values, "values", shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:3]
    return [feature_names[index] for index in top_indices]


def run_xgb_stage() -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Run the Day 1 XGBoost baseline training stage.

    Returns:
        Validation metrics, test metrics, and top three SHAP feature names.
    """
    if not RAW_CREDIT_PATH.exists():
        raise FileNotFoundError(f"Missing required raw dataset: {RAW_CREDIT_PATH}")

    raw_df = pd.read_csv(RAW_CREDIT_PATH)
    clean_df = clean_tabular(raw_df)
    feature_df = engineer_tabular_features(clean_df)
    train_df, val_df, test_df = _split_data(feature_df)
    _save_splits(train_df, val_df, test_df)

    X_train, y_train = _feature_target(train_df)
    X_val, y_val = _feature_target(val_df)
    X_test, y_test = _feature_target(test_df)

    model, val_metrics = train_xgb_baseline(X_train, y_train, X_val, y_val)
    test_prob = predict(model, X_test)
    test_metrics = _metrics(y_test, test_prob)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "xgb_baseline.pkl").open("wb") as file:
        pickle.dump(model, file)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = list(X_train.columns)
    shap_values = plot_shap_summary(
        model,
        X_test,
        feature_names,
        str(FIGURES_DIR / "shap_summary_day1.png"),
    )

    high_risk_index = int(np.argmax(test_prob))
    plot_shap_waterfall(
        shap_values[high_risk_index : high_risk_index + 1],
        X_test.iloc[high_risk_index],
        feature_names,
        str(FIGURES_DIR / "shap_waterfall_day1.png"),
    )

    top_features = _top_shap_features(shap_values, feature_names)
    print(f"[INFO] Val metrics: {val_metrics}")
    print(f"[INFO] Test metrics: {test_metrics}")
    print(f"[INFO] Top 3 SHAP features: {top_features}")
    return val_metrics, test_metrics, top_features


def main() -> None:
    """Parse CLI args and run the requested training stage."""
    parser = argparse.ArgumentParser(description="Credit risk training loop.")
    parser.add_argument("--stage", choices=["xgb"], required=True)
    args = parser.parse_args()

    if args.stage == "xgb":
        run_xgb_stage()


if __name__ == "__main__":
    main()
