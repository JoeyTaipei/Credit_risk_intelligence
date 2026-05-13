"""Second-generation tabular feature engineering for GiveMeSomeCredit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import clean_tabular, engineer_tabular_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CREDIT_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
TARGET_COL = "SeriousDlqin2yrs"

UTIL_COL = "RevolvingUtilizationOfUnsecuredLines"
DELINQ_30_COL = "NumberOfTime30-59DaysPastDueNotWorse"
DELINQ_60_COL = "NumberOfTime60-89DaysPastDueNotWorse"
DELINQ_90_COL = "NumberOfTimes90DaysLate"

V2_FEATURES: list[str] = [
    "util_x_debt",
    "overdue_per_year",
    "income_stability",
    "debt_per_dependent",
    "util_x_overdue",
    "income_rank",
    "util_rank",
    "debt_rank",
    "delinq_acceleration",
    "risk_composite",
]


def engineer_tabular_features_v2(
    df: pd.DataFrame,
    *,
    clean: bool = False,
) -> pd.DataFrame:
    """Return existing engineered tabular features plus the v2 feature set.

    Args:
        df: Raw or already-cleaned GiveMeSomeCredit DataFrame.
        clean: If True, apply ``clean_tabular`` before feature engineering.

    Returns:
        A single DataFrame preserving all columns/features from
        ``engineer_tabular_features`` and adding ``V2_FEATURES``.
    """
    base_df = clean_tabular(df) if clean else df.copy()
    feature_df = engineer_tabular_features(base_df)

    age_decades = feature_df["age"] / 10.0
    age_decades_safe = np.maximum(age_decades, 1.0)
    dependent_count = feature_df["NumberOfDependents"] + 1.0

    weighted_delinquency = (
        feature_df[DELINQ_90_COL] * 3.0
        + feature_df[DELINQ_60_COL] * 2.0
        + feature_df[DELINQ_30_COL]
    )

    feature_df["util_x_debt"] = feature_df[UTIL_COL] * feature_df["DebtRatio"]
    feature_df["overdue_per_year"] = feature_df["total_past_due"] / age_decades_safe
    feature_df["income_stability"] = feature_df["MonthlyIncome"] / dependent_count
    feature_df["debt_per_dependent"] = feature_df["DebtRatio"] / dependent_count
    feature_df["util_x_overdue"] = feature_df[UTIL_COL] * feature_df["total_past_due"]

    feature_df["income_rank"] = feature_df["MonthlyIncome"].rank(pct=True)
    feature_df["util_rank"] = feature_df[UTIL_COL].rank(pct=True)
    feature_df["debt_rank"] = feature_df["DebtRatio"].rank(pct=True)

    feature_df["delinq_acceleration"] = weighted_delinquency / age_decades_safe
    feature_df["risk_composite"] = (
        feature_df["util_x_debt"] + feature_df["delinq_acceleration"]
    )

    return feature_df


def load_credit_features_v2(path: Path | str = RAW_CREDIT_PATH) -> pd.DataFrame:
    """Load raw GiveMeSomeCredit data and return cleaned v2 tabular features."""
    raw_df = pd.read_csv(path)
    return engineer_tabular_features_v2(raw_df, clean=True)


def v2_target_correlations(feature_df: pd.DataFrame) -> pd.Series:
    """Compute Pearson correlations between v2 features and the binary target."""
    if TARGET_COL not in feature_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    return (
        feature_df[V2_FEATURES + [TARGET_COL]]
        .corr(numeric_only=True)[TARGET_COL]
        .drop(labels=TARGET_COL)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )


def main() -> None:
    """Print a quick Step 1 audit of the new feature set."""
    feature_df = load_credit_features_v2()
    correlations = v2_target_correlations(feature_df)

    print("V2 feature list:")
    for feature_name in V2_FEATURES:
        print(f"- {feature_name}")

    print(f"\nRows: {len(feature_df):,}")
    print(f"Columns after v2 engineering: {feature_df.shape[1]}")
    print("\nCorrelation with SeriousDlqin2yrs:")
    print(correlations.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
