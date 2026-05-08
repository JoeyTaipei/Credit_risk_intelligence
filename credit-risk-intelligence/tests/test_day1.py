from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import clean_tabular, engineer_tabular_features
from src.models import xgb_baseline


REQUIRED_COLUMNS = {
    "SeriousDlqin2yrs": 0,
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0.0,
}


def _credit_frame(**overrides: object) -> pd.DataFrame:
    row = {**REQUIRED_COLUMNS, **overrides}
    return pd.DataFrame([row, {**REQUIRED_COLUMNS, "SeriousDlqin2yrs": 1, "age": 55}])


def test_clean_tabular_removes_sentinels() -> None:
    df = _credit_frame(**{"NumberOfTime30-59DaysPastDueNotWorse": 96})

    cleaned = clean_tabular(df)

    assert 96 not in cleaned["NumberOfTime30-59DaysPastDueNotWorse"].to_numpy()


def test_clean_tabular_caps_utilization() -> None:
    df = _credit_frame(RevolvingUtilizationOfUnsecuredLines=5.0)

    cleaned = clean_tabular(df)

    assert cleaned["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.0


def test_engineer_features_adds_columns(sample_credit_df: pd.DataFrame) -> None:
    cleaned = clean_tabular(sample_credit_df)

    engineered = engineer_tabular_features(cleaned)

    expected = {
        "income_per_dependent",
        "total_past_due",
        "has_any_delinquency",
        "credit_line_utilization",
        "debt_to_income_log",
        "age_bucket",
    }
    assert expected.issubset(engineered.columns)


def test_xgb_baseline_returns_metrics(monkeypatch, sample_credit_df: pd.DataFrame) -> None:
    monkeypatch.setitem(xgb_baseline.DEFAULT_XGB_PARAMS, "n_estimators", 10)
    cleaned = clean_tabular(sample_credit_df)
    engineered = engineer_tabular_features(cleaned)
    X = engineered.drop(columns=["SeriousDlqin2yrs"])
    y = engineered["SeriousDlqin2yrs"]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    _, metrics = xgb_baseline.train_xgb_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        use_smote=False,
    )

    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
