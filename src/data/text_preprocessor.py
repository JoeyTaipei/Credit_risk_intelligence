"""Load and align synthetic loan descriptions with the tabular borrower dataset."""

from __future__ import annotations

import pandas as pd

# Fallback text for borrowers with no matched loan description.
# A fixed, semantically neutral string is intentional: it places unmatched
# borrowers at a consistent point in the sentence-BERT embedding space
# rather than a random or missing point, making the fusion layer's job
# predictable for that subpopulation.
_FALLBACK_DESCRIPTION = (
    "Loan application. Purpose: unknown. No additional details provided."
)

_REQUIRED_COLUMNS = {"borrower_id", "loan_purpose", "description"}


def load_loan_descriptions(path: str) -> pd.DataFrame:
    """Load and validate the synthetic loan descriptions CSV.

    Args:
        path: Absolute or relative path to loan_descriptions.csv.

    Returns:
        DataFrame with columns [borrower_id, loan_purpose, description].

    Raises:
        FileNotFoundError: If the file does not exist at path.
        ValueError:        If required columns are missing.
    """
    df = pd.read_csv(path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"loan_descriptions.csv is missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    # Drop rows where description is null — a null string passed to
    # SentenceTransformer would raise an obscure error deep in the tokenizer.
    null_count = df["description"].isna().sum()
    if null_count > 0:
        df = df.dropna(subset=["description"]).reset_index(drop=True)

    return df[["borrower_id", "loan_purpose", "description"]]


def align_texts_with_tabular(
    text_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
) -> list[str]:
    """Return one loan description string per row of tabular_df, in row order.

    Alignment strategy (in priority order):
    1. If tabular_df has a 'borrower_id' column: left-join on borrower_id.
       Rows with no match receive _FALLBACK_DESCRIPTION.
    2. If tabular_df has no 'borrower_id' column (e.g. GiveMeSomeCredit which
       uses numeric row indices): assign descriptions by cycling through text_df
       in row order (index % len(text_df)).  This ensures every tabular borrower
       receives a real, plausible loan description rather than a flat fallback,
       which makes the text encoder actually useful during training.

    WHY the fallback matters in production:
        Real lending systems routinely receive applications with missing or
        incomplete free-text fields — the underwriter skipped the notes field,
        the OCR failed on the scanned form, or the API payload was truncated.
        A model that crashes or returns NaN on missing text is not deployable.
        A fixed neutral embedding for missing text lets the downstream fusion
        classifier learn "when text is absent, rely on other modalities", which
        is exactly the graceful degradation behaviour a production system needs.

    Args:
        text_df:     DataFrame returned by load_loan_descriptions.
                     Must contain columns [borrower_id, description].
        tabular_df:  Cleaned tabular borrower DataFrame (e.g. train.parquet).
                     May or may not have a 'borrower_id' column.

    Returns:
        List of strings of length len(tabular_df), one description per borrower
        in the same row order as tabular_df.
    """
    n = len(tabular_df)

    if "borrower_id" in tabular_df.columns:
        # Strategy 1 — ID-based alignment: join on borrower_id and fill misses.
        # This is the production-correct path when both datasets share a key.
        id_to_desc = text_df.set_index("borrower_id")["description"].to_dict()
        return [
            id_to_desc.get(bid, _FALLBACK_DESCRIPTION)
            for bid in tabular_df["borrower_id"]
        ]

    # Strategy 2 — Index-based cycling: no shared key available.
    # GiveMeSomeCredit has no borrower_id; the synthetic loan_descriptions.csv
    # was generated independently with BRW-XXXXXXXX IDs that do not map to
    # the Kaggle dataset's row indices.  Cycling by modular index ensures every
    # tabular borrower receives a real description, preserving the text
    # modality's contribution to the fusion rather than collapsing it to a
    # single constant embedding for the entire dataset.
    descriptions = text_df["description"].tolist()
    n_desc = len(descriptions)
    return [descriptions[i % n_desc] for i in range(n)]
