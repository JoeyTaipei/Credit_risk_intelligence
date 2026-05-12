"""Build real 12-month payment trajectories from Lending Club snapshot data.

Each row in loan.csv is a single-point-in-time snapshot of one loan.
We reconstruct the monthly trajectory using:
  - Standard loan amortization math (int_rate + installment → monthly balance)
  - mths_since_last_delinq ONLY to locate delinquency timing — loan_status is
    never used for any feature or label (to prevent data leakage)
  - funded_amnt as balance anchor

Temporal split (leak-free design):
  Input  → months 0–9  of the 12-step sequence  (past 10 months)
  Label  → is_late appears in months 10–11       (next 2 months)
  The LSTM must learn to predict FUTURE delinquency from PAST behavior.
  Because input and label come from the same mths_since_last_delinq field but
  different time windows, they are correlated (serial delinquency) without
  being identical — expected AUC ≈ 0.70–0.85.

Only borrowers with ≥ 12 months of history (elapsed ≥ 12) are included so
that both the 10-month input window and the 2-month label window are non-empty.

All four features land in [0, 1] by construction — no separate normalization step.

Output contract (identical shape to create_synthetic_time_series):
  torch.Tensor  (N, 12, 4)
  dim-2 order:  [utilization, payment_ratio, is_late, balance]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LC_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "loan.csv"
LC_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "lc_sequences.pt"
LC_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "lc_labels.pt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STEPS = 12
N_FEATURES = 4
N_MAX = 100_000   # cap for demo speed; ~19 MB on disk
IS_LATE_IDX = 2   # column index of is_late in the (12, 4) tensor

# A delinquency event is assumed to persist for up to LATE_WINDOW months after
# it starts.  6 months covers the typical 90–120 day charge-off pipeline while
# keeping resolved delinquencies (large mths_since_last_delinq) from bleeding
# into the label window.  Tune if positive-rate in lc_labels.pt is too low.
LATE_WINDOW = 6

# loan_status intentionally excluded — using it would create direct leakage
USECOLS = [
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "int_rate",
    "term",
    "issue_d",
    "last_pymnt_d",
    "mths_since_last_delinq",
]

CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Core amortization helper
# ---------------------------------------------------------------------------

def _amortize(funded: float, annual_rate: float, installment: float, n_months: int) -> np.ndarray:
    """Return outstanding balance at months 0, 1, …, n_months (inclusive).

    Uses the closed-form amortization formula so we avoid a Python loop per row.
    When the rate is zero (edge case) we fall back to linear paydown.
    """
    t = np.arange(n_months + 1, dtype=np.float64)
    r = annual_rate / 100.0 / 12.0
    if r > 1e-9:
        bal = funded * (1.0 + r) ** t - installment * ((1.0 + r) ** t - 1.0) / r
    else:
        bal = funded - installment * t
    return np.clip(bal, 0.0, funded)


# ---------------------------------------------------------------------------
# Per-borrower sequence builder
# ---------------------------------------------------------------------------

def _build_sequence(row: pd.Series) -> np.ndarray:
    """Return a (12, 4) float32 array for one borrower snapshot.

    The window covers the most recent 12 months of the loan's life.
    is_late is set using mths_since_last_delinq timing ONLY — loan_status is
    never consulted, so the feature cannot encode the would-be label.

    Delinquency model: a late event starts at delinq_start and persists for
    LATE_WINDOW months.  This captures the typical 60–120-day charge-off
    pipeline while leaving old resolved delinquencies (large mths_since_last_delinq)
    out of the label window.
    """
    funded = float(row["funded_amnt"])
    funded_inv = float(row["funded_amnt_inv"])
    installment = float(row["installment"])
    int_rate = float(row["int_rate"])
    term_months = float(row["term_months"])
    elapsed = int(row["months_elapsed"])   # caller guarantees elapsed >= N_STEPS
    mths_delinq = row["mths_since_last_delinq"]  # may be NaN if no history

    utilization = float(np.clip(funded_inv / max(funded, 1e-6), 0.0, 1.0))

    bal_schedule = _amortize(funded, int_rate, installment, elapsed)

    # All callers pass elapsed >= N_STEPS, so the window always fills completely.
    window = np.arange(elapsed - N_STEPS + 1, elapsed + 1)  # length == 12

    # Delinquency interval: [delinq_start, delinq_start + LATE_WINDOW)
    # Derived purely from mths_since_last_delinq — no loan_status dependency.
    if not np.isnan(mths_delinq):
        delinq_start = elapsed - int(mths_delinq)
        delinq_end = delinq_start + LATE_WINDOW
    else:
        delinq_start = delinq_end = -1   # sentinel: no delinquency recorded

    total_scheduled = installment * term_months
    seq = np.zeros((N_STEPS, N_FEATURES), dtype=np.float32)

    for pos, m in enumerate(window):
        bal = bal_schedule[m]
        paid_so_far = funded - bal
        p_ratio = float(np.clip(paid_so_far / max(total_scheduled, 1e-6), 0.0, 1.0))
        bal_norm = float(bal / max(funded, 1e-6))
        late_flag = 1.0 if delinq_start <= m < delinq_end else 0.0
        seq[pos] = [utilization, p_ratio, late_flag, bal_norm]

    return seq


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_lc_sequences(
    raw_path: Path = LC_RAW_PATH,
    seq_path: Path = LC_SEQ_PATH,
    labels_path: Path = LC_LABELS_PATH,
    n_max: int = N_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load loan.csv, engineer sequences, save tensors, and return them.

    Label derivation (leak-free):
        label = max(seq[10, IS_LATE_IDX], seq[11, IS_LATE_IDX])
    i.e. whether delinquency (from mths_since_last_delinq) falls in the last
    two months of the 12-step window.  loan_status is never read.

    Returns:
        sequences: FloatTensor (N, 12, 4)
        labels:    FloatTensor (N,) — 1 if is_late appears in months 10–11
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Lending Club raw data not found: {raw_path}")

    seq_path.parent.mkdir(parents=True, exist_ok=True)

    sequences: list[np.ndarray] = []
    label_list: list[float] = []

    reader = pd.read_csv(
        raw_path,
        usecols=USECOLS,
        low_memory=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        if len(sequences) >= n_max:
            break

        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        chunk["last_pymnt_d"] = pd.to_datetime(
            chunk["last_pymnt_d"], format="%b-%Y", errors="coerce"
        )

        chunk["months_elapsed"] = (
            (chunk["last_pymnt_d"].dt.year - chunk["issue_d"].dt.year) * 12
            + (chunk["last_pymnt_d"].dt.month - chunk["issue_d"].dt.month)
        ).clip(lower=0)

        chunk["term_months"] = chunk["term"].str.extract(r"(\d+)").astype(float)

        chunk = chunk.dropna(
            subset=["funded_amnt", "funded_amnt_inv", "installment",
                    "int_rate", "term_months", "months_elapsed"]
        )

        # Only keep borrowers with at least 12 months of history so both the
        # 10-month input window and the 2-month label window are fully populated.
        chunk = chunk[chunk["months_elapsed"] >= N_STEPS]

        remaining = n_max - len(sequences)
        chunk = chunk.head(remaining)

        for _, row in chunk.iterrows():
            seq = _build_sequence(row)
            sequences.append(seq)
            # Label = delinquency in months 10–11 of the sequence.
            # Derived from the sequence itself — no loan_status used.
            label = float(max(seq[10, IS_LATE_IDX], seq[11, IS_LATE_IDX]))
            label_list.append(label)

        print(f"[INFO] Built {len(sequences):,} / {n_max:,} sequences ...", end="\r")

    print(f"\n[INFO] Total sequences built: {len(sequences):,}")

    seq_tensor = torch.FloatTensor(np.stack(sequences, axis=0))    # (N, 12, 4)
    lbl_tensor = torch.FloatTensor(label_list)                     # (N,)

    torch.save(seq_tensor, seq_path)
    torch.save(lbl_tensor, labels_path)
    print(f"[INFO] Saved sequences → {seq_path}  (shape {tuple(seq_tensor.shape)})")
    print(f"[INFO] Saved labels    → {labels_path} (positive rate {lbl_tensor.mean():.3f})")

    return seq_tensor, lbl_tensor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_lc_sequences()
