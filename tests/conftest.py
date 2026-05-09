from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_credit_df() -> pd.DataFrame:
    """Return a reproducible 100-row GiveMeSomeCredit-style sample DataFrame."""
    rng = np.random.default_rng(seed=42)
    row_count = 100

    monthly_income = rng.normal(loc=6500, scale=2500, size=row_count).clip(min=0)
    dependents = rng.integers(0, 6, size=row_count).astype(float)

    # Inject 5% missingness into nullable competition fields.
    missing_indices = rng.choice(row_count, size=5, replace=False)
    monthly_income[missing_indices] = np.nan
    dependents[rng.choice(row_count, size=5, replace=False)] = np.nan

    return pd.DataFrame(
        {
            "SeriousDlqin2yrs": rng.integers(0, 2, size=row_count),
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2, size=row_count),
            "age": rng.integers(21, 91, size=row_count),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 11, size=row_count),
            "DebtRatio": rng.uniform(0, 3, size=row_count),
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 25, size=row_count),
            "NumberOfTimes90DaysLate": rng.integers(0, 11, size=row_count),
            "NumberRealEstateLoansOrLines": rng.integers(0, 8, size=row_count),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 11, size=row_count),
            "NumberOfDependents": dependents,
        }
    )


@pytest.fixture
def sample_text_corpus() -> list[str]:
    """Return ten varied dummy loan descriptions for text-pipeline tests."""
    return [
        "I need a loan to consolidate several credit card balances. A single payment would help me manage cash flow.",
        "The loan will fund kitchen repairs and new flooring. I plan to repay it over three years.",
        "I am requesting funds for unexpected dental treatment. My income is stable and repayment is budgeted.",
        "This loan would support inventory purchases for my small business. Sales usually rise during the holiday season.",
        "I need short-term financing for moving expenses. The relocation is tied to a new job offer.",
        "The funds will cover vehicle repairs so I can commute to work. I expect to repay monthly from salary.",
        "I am applying for a medical loan after a recent hospital visit. Insurance covered part of the bill.",
        "The loan would help upgrade business equipment. The new machine should improve monthly revenue.",
        "I want to refinance older high-interest debt. Lower payments would reduce pressure on my budget.",
        "I need support for education costs this semester. My repayment plan starts after tuition is paid.",
    ]
