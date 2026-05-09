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


def create_synthetic_time_series(
    df: pd.DataFrame,
    window: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """Simulate monthly payment sequences from GiveMeSomeCredit tabular features.

    Each borrower gets a (window, 4) sequence of monthly observations:
        [utilization_t, payment_ratio_t, is_late_t, balance_t]

    The four tabular columns used as generative seeds:
        - RevolvingUtilizationOfUnsecuredLines → base monthly credit utilization
        - NumberOfTime30-59DaysPastDueNotWorse → per-month late-payment probability
        - DebtRatio → scales the outstanding balance proxy
        - MonthlyIncome → normalises the balance proxy (high income = lower stress)

    Args:
        df: Cleaned borrower DataFrame (output of clean_tabular).
        window: Number of synthetic monthly timesteps per borrower.
        seed: NumPy random seed for reproducibility.

    Returns:
        Float32 array of shape (n_borrowers, window, 4), each feature in [0, 1].
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    n = len(df)

    # --- Seed features (all already cleaned and bounded by clean_tabular) ---
    # Fill missing seed values defensively so raw fixture data does not
    # propagate NaNs into tensors.
    util_seed = df["RevolvingUtilizationOfUnsecuredLines"].fillna(
        df["RevolvingUtilizationOfUnsecuredLines"].median()
    )
    late_seed = df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(
        df["NumberOfTime30-59DaysPastDueNotWorse"].median()
    )
    debt_seed = df["DebtRatio"].fillna(df["DebtRatio"].median())
    income_seed = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # RevolvingUtilization is already capped to [0, 1] — use directly as the
    # borrower's baseline monthly credit usage level.
    util_base = util_seed.values.astype(np.float32)

    # Convert 30-59 day late count to a per-month probability.  A borrower who
    # was late 3 times in a 12-month window has ~25% probability of being late
    # in any given simulated month.  Cap at 0.5 so even the worst borrowers
    # still have on-time months — preserving variation in payment_ratio_t.
    late_prob = np.clip(
        late_seed.values.astype(np.float32) / window,
        0.0,
        0.5,
    )  # shape: (n,)

    # DebtRatio and MonthlyIncome are combined into a balance-stress proxy.
    # Normalise each to [0, 1] so neither dominates the other.
    debt = debt_seed.values.astype(np.float32)
    debt_norm = (debt - debt.min()) / (debt.max() - debt.min() + 1e-8)

    income = income_seed.values.astype(np.float32)
    income_norm = (income - income.min()) / (income.max() - income.min() + 1e-8)

    # --- Feature 0: utilization_t ---
    # Base utilization plus a small Gaussian random walk each month.
    # High-utilization borrowers stay high on average; noise adds month-to-month
    # variation without changing the underlying credit behaviour signal.
    util_noise = rng.normal(0.0, 0.05, size=(n, window)).astype(np.float32)
    utilization_t = np.clip(util_base[:, np.newaxis] + util_noise, 0.0, 1.0)

    # --- Feature 2: is_late_t  (computed before payment_ratio to condition it) ---
    # Bernoulli draw per borrower per month using their individual late_prob.
    # Borrowers with more historical delinquencies have higher monthly late probability.
    is_late_t = (
        rng.random(size=(n, window)).astype(np.float32) < late_prob[:, np.newaxis]
    ).astype(np.float32)  # binary {0.0, 1.0}

    # --- Feature 1: payment_ratio_t ---
    # On-time months: payment ~95% of minimum due (good payer, slight variation).
    # Late months:    payment ~50% of minimum due (partial payment, financial stress).
    # The 0.45 gap encodes the business intuition that late payments co-occur
    # with lower payment amounts — they are not independent events.
    pay_noise = rng.normal(0.0, 0.08, size=(n, window)).astype(np.float32)
    pay_base = 0.95 - 0.45 * is_late_t  # (n, window)
    payment_ratio_t = np.clip(pay_base + pay_noise, 0.0, 1.0)

    # --- Feature 3: balance_t ---
    # High DebtRatio and low MonthlyIncome jointly signal a borrower carrying
    # a heavier outstanding balance relative to their means.  Multiply
    # debt_norm by (1 - 0.5 * income_norm) so that higher income attenuates
    # the balance stress rather than eliminating it entirely.
    balance_base = debt_norm * (1.0 - 0.5 * income_norm)  # (n,)
    balance_noise = rng.normal(0.0, 0.05, size=(n, window)).astype(np.float32)
    balance_t = np.clip(balance_base[:, np.newaxis] + balance_noise, 0.0, 1.0)

    # Stack to (n, window, 4) in the documented feature order.
    sequences = np.stack(
        [utilization_t, payment_ratio_t, is_late_t, balance_t], axis=2
    )  # (n, window, 4)

    # Final per-feature min-max normalisation across the entire dataset so the
    # LSTM receives consistent [0, 1] inputs regardless of distributional shift
    # between batches.  Skips features whose range is already collapsed (e.g.
    # is_late_t is already {0, 1}).
    for f in range(sequences.shape[2]):
        feat = sequences[:, :, f]
        f_min, f_max = feat.min(), feat.max()
        if f_max > f_min:
            sequences[:, :, f] = (feat - f_min) / (f_max - f_min)

    return sequences  # (n_borrowers, window, 4), dtype float32


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
