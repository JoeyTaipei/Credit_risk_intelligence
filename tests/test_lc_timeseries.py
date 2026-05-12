"""Validate lc_sequences.pt shape, leakage absence, and is_late serial correlation.

Run after building sequences:
    python -m src.data.lending_club_timeseries
    pytest tests/test_lc_timeseries.py -v
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "lc_sequences.pt"
PLOT_PATH = PROJECT_ROOT / "data" / "processed" / "lc_sequence_sample.png"

FEATURE_NAMES = ["utilization", "payment_ratio", "is_late", "balance"]
IS_LATE_IDX = 2  # column index of is_late in the (12, 4) tensor


@pytest.fixture(scope="module")
def sequences() -> torch.Tensor:
    if not SEQ_PATH.exists():
        pytest.skip(f"lc_sequences.pt not found at {SEQ_PATH}. Run src/data/lending_club_timeseries.py first.")
    return torch.load(SEQ_PATH, map_location="cpu")


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------

def test_tensor_is_three_dimensional(sequences: torch.Tensor) -> None:
    assert sequences.ndim == 3, f"Expected 3D tensor, got {sequences.ndim}D"


def test_sequence_length_is_12(sequences: torch.Tensor) -> None:
    assert sequences.shape[1] == 12, f"Expected 12 timesteps, got {sequences.shape[1]}"


def test_feature_count_is_4(sequences: torch.Tensor) -> None:
    assert sequences.shape[2] == 4, f"Expected 4 features, got {sequences.shape[2]}"


def test_at_least_1000_borrowers(sequences: torch.Tensor) -> None:
    assert sequences.shape[0] >= 1_000, f"Too few borrowers: {sequences.shape[0]}"


def test_all_features_in_unit_range(sequences: torch.Tensor) -> None:
    lo = sequences.min().item()
    hi = sequences.max().item()
    assert lo >= -1e-5, f"Feature below 0: min={lo:.6f}"
    assert hi <= 1.0 + 1e-5, f"Feature above 1: max={hi:.6f}"


def test_is_late_is_binary(sequences: torch.Tensor) -> None:
    is_late = sequences[:, :, IS_LATE_IDX]
    unique_vals = is_late.unique().tolist()
    non_binary = [v for v in unique_vals if abs(v) > 1e-5 and abs(v - 1.0) > 1e-5]
    assert not non_binary, f"is_late has non-binary values: {non_binary[:5]}"


# ---------------------------------------------------------------------------
# Leakage guard: months 10-11 must not be a trivial copy of months 0-9
# ---------------------------------------------------------------------------

def test_no_perfect_input_to_label_leakage(sequences: torch.Tensor) -> None:
    """The label (is_late in months 10-11) must not be perfectly predictable
    from is_late in the input (months 0-9 only).

    If is_late[:,9] == label for every borrower, the temporal split is broken
    and the LSTM can achieve AUC=1 by copying a single input feature.
    """
    lbl_path = SEQ_PATH.parent / "lc_labels.pt"
    if not lbl_path.exists():
        pytest.skip("lc_labels.pt not found")

    labels = torch.load(lbl_path, map_location="cpu")
    last_input_late = sequences[:, 9, IS_LATE_IDX]  # month 9 (last input month)

    # Count borrowers where the label disagrees with the last input is_late flag
    disagreements = (labels != last_input_late).float().mean().item()
    assert disagreements > 0.05, (
        f"Only {disagreements*100:.1f}% disagreement between input[month-9].is_late "
        f"and label — likely still leaking.  Expected >5%."
    )


# ---------------------------------------------------------------------------
# Serial correlation: consecutive late payments should cluster together
# ---------------------------------------------------------------------------

def test_serial_correlation_in_is_late(sequences: torch.Tensor) -> None:
    """Late payments must exhibit positive autocorrelation across timesteps.

    For each borrower who has at least one late flag, check that consecutive
    months are more likely to both be late than a random pair would be.
    """
    is_late = sequences[:, :, IS_LATE_IDX].numpy()  # (N, 12)

    # Only look at borrowers who have at least one late month
    late_mask = is_late.max(axis=1) > 0
    late_seqs = is_late[late_mask]

    assert len(late_seqs) > 0, "No borrowers with any late payments found"

    # Consecutive-pair autocorrelation: P(t+1 late | t late) vs P(t+1 late | t ok)
    t_vals = late_seqs[:, :-1].ravel()   # months 0..10
    t1_vals = late_seqs[:, 1:].ravel()  # months 1..11

    p_late_given_late = t1_vals[t_vals == 1].mean() if (t_vals == 1).any() else 0.0
    p_late_given_ok = t1_vals[t_vals == 0].mean() if (t_vals == 0).any() else 0.0

    print(f"\n  P(late_t+1 | late_t) = {p_late_given_late:.3f}")
    print(f"  P(late_t+1 | ok_t)   = {p_late_given_ok:.3f}")

    # Serial correlation means a late month predicts the next one
    assert p_late_given_late > p_late_given_ok, (
        f"No serial correlation detected: "
        f"P(late|late)={p_late_given_late:.3f} <= P(late|ok)={p_late_given_ok:.3f}"
    )


# ---------------------------------------------------------------------------
# Visualization: 3 sample borrowers
# ---------------------------------------------------------------------------

def test_plot_sample_borrowers(sequences: torch.Tensor) -> None:
    """Plot 3 borrowers with different risk profiles and save to disk."""
    is_late_col = sequences[:, :, IS_LATE_IDX]

    # Find three representative borrowers:
    # 1. A borrower who was never late
    # 2. A borrower who became late mid-sequence
    # 3. A borrower who was late throughout
    never_late = (is_late_col.max(dim=1).values == 0).nonzero(as_tuple=True)[0]
    mid_late = (
        (is_late_col[:, :6].max(dim=1).values == 0)
        & (is_late_col[:, 6:].max(dim=1).values == 1)
    ).nonzero(as_tuple=True)[0]
    always_late = (is_late_col.min(dim=1).values == 1).nonzero(as_tuple=True)[0]

    # Fall back to first borrower with any late flag if ideal case not found
    any_late = (is_late_col.max(dim=1).values == 1).nonzero(as_tuple=True)[0]

    candidates = [
        ("Never Late",  never_late[0].item() if len(never_late) else 0),
        ("Becomes Late", mid_late[0].item() if len(mid_late) else (any_late[0].item() if len(any_late) else 1)),
        ("Always Late",  always_late[0].item() if len(always_late) else (any_late[-1].item() if len(any_late) else 2)),
    ]

    months = np.arange(1, 13)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Lending Club: 3 Sample Borrower Sequences", fontsize=13, fontweight="bold")

    for ax, (title, idx) in zip(axes, candidates):
        seq = sequences[idx].numpy()  # (12, 4)
        ax2 = ax.twinx()
        ax.bar(months, seq[:, IS_LATE_IDX], color="crimson", alpha=0.4, label="is_late")
        ax2.plot(months, seq[:, 1], "steelblue", marker="o", ms=4, label="payment_ratio")
        ax2.plot(months, seq[:, 3], "darkorange", marker="s", ms=4, label="balance")
        ax.set_ylim(-0.05, 1.3)
        ax2.set_ylim(-0.05, 1.3)
        ax.set_ylabel("is_late", color="crimson", fontsize=9)
        ax2.set_ylabel("ratio [0,1]", fontsize=9)
        ax.set_title(f"Borrower #{idx} — {title}", fontsize=10)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Month", fontsize=10)
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()

    assert PLOT_PATH.exists(), f"Plot was not saved to {PLOT_PATH}"
    print(f"\n  Plot saved → {PLOT_PATH}")
