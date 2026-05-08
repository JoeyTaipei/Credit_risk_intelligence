from __future__ import annotations

import pandas as pd


EXPECTED_CREDIT_COLUMNS = {
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}


def test_train_loop_imports() -> None:
    import src.training.train_loop

    assert src.training.train_loop is not None


def test_preprocess_imports() -> None:
    import src.data.preprocess

    assert src.data.preprocess is not None


def test_sample_credit_df_fixture_has_expected_columns(sample_credit_df: pd.DataFrame) -> None:
    assert EXPECTED_CREDIT_COLUMNS.issubset(sample_credit_df.columns)
