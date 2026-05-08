from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.text_preprocessor import align_texts_with_tabular, load_loan_descriptions
from src.models.text_encoder import TextEncoder


def test_text_encoder_output_shape() -> None:
    encoder = TextEncoder(freeze=True)
    texts = [
        "Debt consolidation request with stable salary.",
        "Home improvement loan for kitchen renovation.",
        "Medical expense loan after a hospital visit.",
        "Small business working capital request.",
        "Personal loan for moving costs.",
    ]

    output = encoder.encode_texts(texts)

    assert output.shape == (5, 32)


def test_text_encoder_frozen() -> None:
    encoder = TextEncoder(freeze=True)

    assert all(not param.requires_grad for param in encoder.sentence_transformer.parameters())


def test_align_texts_fallback() -> None:
    text_df = pd.DataFrame(
        {
            "borrower_id": ["BRW-001"],
            "loan_purpose": ["other"],
            "description": ["Known borrower description."],
        }
    )
    tabular_df = pd.DataFrame({"borrower_id": ["BRW-001", "BRW-404"]})

    aligned = align_texts_with_tabular(text_df, tabular_df)

    assert len(aligned) == len(tabular_df)
    assert "No additional details provided" in aligned[1]


def test_load_descriptions_columns() -> None:
    path = Path("data/synthetic/loan_descriptions.csv")

    df = load_loan_descriptions(str(path))

    assert {"borrower_id", "loan_purpose", "description"}.issubset(df.columns)
