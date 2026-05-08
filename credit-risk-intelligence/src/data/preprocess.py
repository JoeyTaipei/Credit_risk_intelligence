"""Preprocessing pipeline for the Multi-Modal Credit Risk Intelligence System."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Delinquency columns referenced by both clean_tabular and engineer_tabular_features.
_DELINQUENCY_COLS: list[str] = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def clean_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw GiveMeSomeCredit tabular data.

    Transformations applied in order:
    1. Drop unnamed Kaggle index column if present.
    2. Replace sentinel codes 96/98 with NaN in delinquency columns.
    3. Remove rows where age == 0 (impossible value; data entry error).
    4. Cap RevolvingUtilizationOfUnsecuredLines at 1.0 (ratio by definition ≤ 1).
    5. Cap DebtRatio at its 99th percentile (extreme right tail likely errors).
    6. Impute MonthlyIncome with median (heavy right skew; mean is inflated).
    7. Impute NumberOfDependents with 0 (mode; most borrowers list no dependents).
    8. Impute sentinel-cleared delinquency columns with their median (≈ 0).

    Args:
        df: Raw GiveMeSomeCredit DataFrame as loaded from cs-training.csv.

    Returns:
        Cleaned DataFrame with the same column set as input.
    """
    df = df.copy()  # never mutate the caller's DataFrame in place

    # Step 1 — Kaggle exports an unnamed positional index column when the file
    # is loaded without index_col=0.  Drop it if present to avoid a spurious
    # numeric feature leaking into the model.
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Step 2 — 96 and 98 are credit-bureau sentinel codes meaning "unknown /
    # not applicable", not genuine delinquency counts.  Keeping them as-is
    # would make the model treat some borrowers as having been late 96 times,
    # completely corrupting those features.  Replace with NaN before imputing.
    for col in _DELINQUENCY_COLS:
        df[col] = df[col].replace({96: np.nan, 98: np.nan})

    # Step 3 — age == 0 is a data entry error (one row in the dataset).
    # Imputing zero would be arbitrary; removing is safe given the negligible
    # row count and the fact that no real borrower is 0 years old.
    df = df[df["age"] != 0].reset_index(drop=True)

    # Step 4 — RevolvingUtilizationOfUnsecuredLines is a ratio (credit used /
    # credit available).  Values above 1.0 represent data entry errors or
    # non-standard accounting conventions; cap at 1.0 instead of deleting rows
    # so the rest of each borrower's information is preserved.
    df["RevolvingUtilizationOfUnsecuredLines"] = (
        df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.0)
    )

    # Step 5 — DebtRatio has an extreme right tail (max in the millions) caused
    # by near-zero denominator edge cases or data errors.  The 99th-percentile
    # cap removes the most damaging outliers while retaining variation across
    # the vast majority of the distribution.
    debt_cap = df["DebtRatio"].quantile(0.99)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=debt_cap)

    # Step 6 — MonthlyIncome is missing ~20% of rows and is heavily right-skewed.
    # Mean imputation would be inflated by high-income outliers; median is robust
    # to the long tail and better represents the typical borrower's income.
    income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)

    # Step 7 — NumberOfDependents is missing ~2.5% of rows.  The mode is 0
    # (the majority of borrowers list no dependents), making 0 both the
    # statistically and intuitively correct fill value.
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Step 8 — After sentinel replacement in Step 2, the delinquency columns
    # contain NaN where 96/98 previously sat.  The median of each column is
    # effectively 0 (most borrowers have no late payments), so median imputation
    # is conservative and unlikely to introduce bias.
    for col in _DELINQUENCY_COLS:
        df[col] = df[col].fillna(df[col].median())

    return df


def create_synthetic_time_series(df: pd.DataFrame, window: int = 12) -> np.ndarray:
    """Create synthetic borrower time-series windows from tabular data.

    Args:
        df: Borrower credit DataFrame.
        window: Number of synthetic time steps to create per borrower.

    Returns:
        Synthetic time-series feature array.
    """
    raise NotImplementedError("Day 2 deliverable")


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer model-ready tabular features from a cleaned GiveMeSomeCredit DataFrame.

    Adds six columns (originals are preserved):
    - income_per_dependent: income normalised by household size
    - total_past_due: aggregate delinquency count across all three severity buckets
    - has_any_delinquency: binary flag; 1 if borrower has any recorded late payment
    - credit_line_utilization: readability alias for RevolvingUtilizationOfUnsecuredLines
    - debt_to_income_log: log1p-transformed DebtRatio (compresses the right tail)
    - age_bucket: ordinal age band (0=under 30, 1=30–44, 2=45–59, 3=60+)

    Args:
        df: Cleaned borrower credit DataFrame (output of clean_tabular).

    Returns:
        DataFrame with original columns plus the six engineered features.
    """
    df = df.copy()

    # income_per_dependent — normalises income by household size.
    # Adding 1 avoids division by zero and treats a single borrower
    # as a "household of 1", preserving meaningful variation at 0 dependents.
    # High income relative to household size is a strong creditworthiness signal.
    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # total_past_due — aggregates all three delinquency severity buckets.
    # Summing rather than taking the max retains frequency information;
    # a borrower late once is different from one late ten times.
    df["total_past_due"] = df[_DELINQUENCY_COLS].sum(axis=1)

    # has_any_delinquency — binary indicator of any prior late payment.
    # Even a single past-due event is a strong differentiator between clean
    # and risky borrowers, independent of severity or count.
    df["has_any_delinquency"] = (df["total_past_due"] > 0).astype(int)

    # credit_line_utilization — human-readable alias; keeps the original column
    # intact for backward compatibility with downstream modules.
    df["credit_line_utilization"] = df["RevolvingUtilizationOfUnsecuredLines"]

    # debt_to_income_log — log1p compresses DebtRatio's extreme right tail into
    # a scale where tree splits and linear models operate more effectively.
    # log1p (not log) handles zero values without producing -inf.
    df["debt_to_income_log"] = np.log1p(df["DebtRatio"])

    # age_bucket — ordinal bands capture non-linear age–risk relationships.
    # Clipping at 100 keeps values inside the bin range; ages above 100 in the
    # raw data are likely entry errors.  Labels are integers so models treat
    # the variable as ordinal rather than nominal.
    df["age_bucket"] = pd.cut(
        df["age"].clip(upper=100),
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


def build_borrower_graph(df: pd.DataFrame, threshold: float = 0.85) -> tuple:
    """Build borrower relationship graph inputs.

    Args:
        df: Borrower feature DataFrame.
        threshold: Similarity threshold for connecting borrower nodes.

    Returns:
        Tuple containing graph artifacts.
    """
    raise NotImplementedError("Day 4 deliverable")
