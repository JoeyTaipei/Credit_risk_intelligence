"""Generate aligned LSTM embeddings for cs-training2 borrowers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.features_v2 import TARGET_COL, load_credit_features_v2
from src.data.preprocess import create_synthetic_time_series
from src.models.lstm_encoder import LSTMEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training2.csv"
DEFAULT_LSTM_PATH = PROCESSED_DIR / "lstm_encoder_lc.pt"
BATCH_SIZE = 10_000


def _safe_torch_save(tensor: torch.Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    torch.save(tensor, path)


def generate_lstm_embeddings(
    data_path: Path = DEFAULT_DATA_PATH,
    lstm_path: Path = DEFAULT_LSTM_PATH,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, ...]:
    """Encode synthetic aligned sequences with the LC-trained LSTM encoder."""
    output_path = PROCESSED_DIR / "lstm_emb_aligned.pt"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing LSTM checkpoint: {lstm_path}")

    feature_df = load_credit_features_v2(data_path)
    labels = feature_df[TARGET_COL].astype(int)
    sequences = create_synthetic_time_series(feature_df, seed=42).astype(np.float32)
    if len(sequences) != len(labels):
        raise ValueError(f"Sequence count {len(sequences)} does not match labels {len(labels)}")

    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    encoder.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    encoder.eval()

    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch = torch.as_tensor(sequences[start:end], dtype=torch.float32)
            embeddings.append(encoder(batch).detach().cpu())

    output = torch.cat(embeddings, dim=0).float()
    _safe_torch_save(output, output_path)
    print(f"[INFO] Saved {output_path} shape={tuple(output.shape)}")
    print("[INFO] LSTM limitation: synthetic cs-training2 sequences encoded with LC-trained weights.")
    return tuple(output.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lstm-path", type=Path, default=DEFAULT_LSTM_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_lstm_embeddings(
        data_path=args.data_path,
        lstm_path=args.lstm_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
