"""Generate aligned synthetic text embeddings for cs-training2 borrowers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.features_v2 import TARGET_COL, load_credit_features_v2
from src.models.text_encoder import TextEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training2.csv"
DEFAULT_TEXT_ENCODER_PATH = PROCESSED_DIR / "text_encoder.pt"
BATCH_SIZE = 10_000


def _safe_torch_save(tensor: torch.Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    torch.save(tensor, path)


def _money(value: float) -> str:
    return f"${int(max(value, 0)):,}"


def _make_descriptions(feature_df) -> list[str]:
    """Create feature-conditioned synthetic loan descriptions."""
    descriptions: list[str] = []
    for idx, row in feature_df.reset_index(drop=True).iterrows():
        income = float(row["MonthlyIncome"])
        debt_ratio = float(row["DebtRatio"])
        util = float(row["RevolvingUtilizationOfUnsecuredLines"])
        dependents = int(row["NumberOfDependents"])
        open_lines = int(row["NumberOfOpenCreditLinesAndLoans"])
        past_due = int(row["total_past_due"])
        amount = min(max(income * (0.15 + debt_ratio * 0.05), 1000), 40000)

        if past_due >= 2 or util > 0.8:
            purpose = "debt consolidation"
            reason = "to consolidate revolving balances and stabilize monthly payments"
        elif dependents >= 2:
            purpose = "family expenses"
            reason = "to manage household expenses and planned family obligations"
        elif debt_ratio > 0.6:
            purpose = "home improvement"
            reason = "to complete essential repairs while keeping payments predictable"
        elif idx % 5 == 0:
            purpose = "medical expenses"
            reason = "to cover unexpected medical bills"
        else:
            purpose = "personal loan"
            reason = "to cover planned personal expenses"

        descriptions.append(
            "I am applying for a {amount} {purpose}. The funds will be used {reason}. "
            "My monthly income is {income}, I have {open_lines} open credit lines, "
            "current debt ratio is {debt_ratio:.3f}, utilization is {util:.3f}, "
            "and I report {dependents} dependents with {past_due} recent past-due events.".format(
                amount=_money(amount),
                purpose=purpose,
                reason=reason,
                income=_money(income),
                open_lines=open_lines,
                debt_ratio=debt_ratio,
                util=util,
                dependents=dependents,
                past_due=past_due,
            )
        )
    return descriptions


def generate_text_embeddings(
    data_path: Path = DEFAULT_DATA_PATH,
    text_encoder_path: Path = DEFAULT_TEXT_ENCODER_PATH,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, ...]:
    """Encode aligned synthetic descriptions with TextEncoder."""
    output_path = PROCESSED_DIR / "text_emb_aligned.pt"
    metadata_path = PROCESSED_DIR / "text_emb_aligned_metadata.json"
    for path in [output_path, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    feature_df = load_credit_features_v2(data_path)
    labels = feature_df[TARGET_COL].astype(int)
    descriptions = _make_descriptions(feature_df)
    if len(descriptions) != len(labels):
        raise ValueError(f"Description count {len(descriptions)} does not match labels {len(labels)}")

    torch.manual_seed(42)
    encoder = TextEncoder(freeze=True)
    loaded_checkpoint = False
    if text_encoder_path.exists():
        encoder.load_state_dict(torch.load(text_encoder_path, map_location="cpu"))
        loaded_checkpoint = True
    else:
        print(
            f"[WARN] Missing {text_encoder_path}; using deterministic freshly initialized "
            "TextEncoder projection head."
        )
    encoder.eval()

    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            end = min(start + batch_size, len(descriptions))
            chunks.append(encoder.encode_texts(descriptions[start:end]).detach().cpu().float())
            print(f"[INFO] Encoded text rows {start:,}-{end:,}")

    output = torch.cat(chunks, dim=0).float()
    if output.shape != (len(descriptions), 32):
        raise ValueError(f"Unexpected text embedding shape: {tuple(output.shape)}")

    _safe_torch_save(output, output_path)
    metadata = {
        "data_path": str(data_path),
        "n_borrowers": int(len(descriptions)),
        "embedding_dim": 32,
        "text_encoder_path": str(text_encoder_path),
        "loaded_checkpoint": loaded_checkpoint,
        "real_or_synthetic": "Synthetic template generated from cs-training2 features",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {output_path} shape={tuple(output.shape)}")
    return tuple(output.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--text-encoder-path", type=Path, default=DEFAULT_TEXT_ENCODER_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_text_embeddings(
        data_path=args.data_path,
        text_encoder_path=args.text_encoder_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
