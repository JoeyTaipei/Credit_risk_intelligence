"""Clean the Power BI dashboard CSV into a stable report-ready schema."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "powerbi_dashboard.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "powerbi_dashboard_clean.csv"

FINAL_COLUMNS = [
    "borrower_id",
    "age",
    "age_bucket",
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTimes90DaysLate",
    "risk_score",
    "risk_score_pct",
    "risk_level",
    "risk_level_en",
    "top_shap_feature",
    "shap_value_top1",
    "recommended_action",
    "is_thin_file",
    "estimated_loss",
    "genai_summary",
]

RENAME_MAP = {
    "total_past_due": "NumberOfTimes90DaysLate",
    "shap_value_1": "shap_value_top1",
    "ai_summary": "genai_summary",
}

RISK_LEVEL_EN = {
    "低風險": "Low",
    "中風險": "Medium",
    "高風險": "High",
}

VALIDATE_NOT_NULL = [
    "borrower_id",
    "risk_score",
    "risk_level",
    "top_shap_feature",
    "recommended_action",
]

ROUNDING = {
    "MonthlyIncome": 2,
    "DebtRatio": 4,
    "RevolvingUtilizationOfUnsecuredLines": 4,
    "risk_score": 4,
    "risk_score_pct": 4,
    "shap_value_top1": 4,
    "estimated_loss": 2,
}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _resolve_paths() -> tuple[Path, Path]:
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()
    elif Path("powerbi_dashboard.csv").exists():
        input_path = Path("powerbi_dashboard.csv").resolve()
    else:
        input_path = DEFAULT_INPUT

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2]).resolve()
    elif input_path.name == "powerbi_dashboard.csv":
        output_path = input_path.with_name("powerbi_dashboard_clean.csv")
    else:
        output_path = DEFAULT_OUTPUT

    return input_path, output_path


def _thin_file_to_int(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"true", "1", "yes", "y"} else 0


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_powerbi_dashboard(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    original_shape = df.shape

    df = df.rename(columns=RENAME_MAP)

    required_columns = [
        "borrower_id",
        "age",
        "age_bucket",
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "risk_score",
        "risk_level",
        "top_shap_feature",
        "shap_value_top1",
        "recommended_action",
        "is_thin_file",
        "genai_summary",
    ]
    _require_columns(df, required_columns)

    df["risk_level"] = df["risk_level"].astype(str).str.strip()
    unknown_risk_levels = sorted(set(df["risk_level"].dropna()) - set(RISK_LEVEL_EN))
    if unknown_risk_levels:
        raise ValueError(f"Unexpected risk_level values: {unknown_risk_levels}")

    clean = df.copy()
    clean["risk_score"] = pd.to_numeric(clean["risk_score"], errors="coerce")
    clean["MonthlyIncome"] = pd.to_numeric(clean["MonthlyIncome"], errors="coerce")
    clean["DebtRatio"] = pd.to_numeric(clean["DebtRatio"], errors="coerce")
    clean["RevolvingUtilizationOfUnsecuredLines"] = pd.to_numeric(
        clean["RevolvingUtilizationOfUnsecuredLines"],
        errors="coerce",
    )
    clean["NumberOfTimes90DaysLate"] = pd.to_numeric(
        clean["NumberOfTimes90DaysLate"],
        errors="coerce",
    ).fillna(0).astype(int)
    clean["shap_value_top1"] = pd.to_numeric(clean["shap_value_top1"], errors="coerce").fillna(0)

    clean["risk_score_pct"] = clean["risk_score"]
    clean["risk_level_en"] = clean["risk_level"].map(RISK_LEVEL_EN)
    clean["is_thin_file"] = clean["is_thin_file"].map(_thin_file_to_int).astype(int)
    clean["estimated_loss"] = 0.0
    high_risk_mask = clean["risk_level"] == "高風險"
    clean.loc[high_risk_mask, "estimated_loss"] = (
        clean.loc[high_risk_mask, "risk_score"] * clean.loc[high_risk_mask, "MonthlyIncome"] * 12
    )
    clean["genai_summary"] = clean["genai_summary"].fillna("無 AI 摘要")
    clean.loc[clean["genai_summary"].astype(str).str.strip() == "", "genai_summary"] = "無 AI 摘要"

    for column, digits in ROUNDING.items():
        clean[column] = clean[column].round(digits)

    clean = clean[FINAL_COLUMNS]

    null_counts = clean[VALIDATE_NOT_NULL].isna().sum()
    failed = null_counts[null_counts > 0]
    if not failed.empty:
        raise ValueError(f"Missing values found in required dashboard columns: {failed.to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {clean.shape}")
    print(f"Final column list: {list(clean.columns)}")
    print("\nrisk_level value counts:")
    print(clean["risk_level"].value_counts(dropna=False).to_string())
    print("\nFirst 5 rows:")
    print(clean.head(5).to_string(index=False))
    print(f"\nWrote: {output_path}")

    return clean


def main() -> None:
    _configure_stdout()
    input_path, output_path = _resolve_paths()
    clean_powerbi_dashboard(input_path, output_path)


if __name__ == "__main__":
    main()
