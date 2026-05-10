"""Regenerate docs/figures with correct Unicode arrows and Chinese labels.

Fixes:
  architecture_overview.png  -- ? glyphs were broken Unicode arrows
  risk_scoring_table.png     -- ??? cells were Chinese risk levels without CJK font
  shap_summary_day1.png      -- regenerate with Traditional Chinese feature names
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "docs" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"

# Microsoft JhengHei ships with Windows and covers Traditional Chinese + full Latin.
CJK_FONT = "Microsoft JhengHei"

# Traditional Chinese labels for every XGBoost feature
FEATURE_ZH = {
    "RevolvingUtilizationOfUnsecuredLines": "循環信用使用率",
    "age_bucket":                           "年齡區段",
    "DebtRatio":                            "負債比率",
    "age":                                  "年齡",
    "debt_to_income_log":                   "負債收入比（對數）",
    "income_per_dependent":                 "每人口平均收入",
    "total_past_due":                       "累計逾期次數",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89天逾期次數",
    "MonthlyIncome":                        "每月收入",
    "NumberOfOpenCreditLinesAndLoans":      "信用帳戶數",
    "NumberRealEstateLoansOrLines":         "不動產貸款數",
    "NumberOfRealEstateLoansOrLines":       "不動產貸款數",
    "NumberOfDependents":                   "依存人口數",
    "NumberOfTimes90DaysLate":              "90天以上逾期次數",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59天逾期次數",
    "credit_line_utilization":              "信用額度使用率",
    "has_any_delinquency":                  "曾有逾期紀錄",
}


# ---------------------------------------------------------------------------
# 1. Architecture overview
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, sub="", facecolor="#4A90D9", textcolor="white",
         fontsize=11, subfontsize=8.5):
    """Draw a rounded rectangle with centred text + optional subtitle."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015",
        facecolor=facecolor,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.012 if sub else 0)
    ax.text(x + w / 2, cy, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub,
                ha="center", va="center",
                fontsize=subfontsize, color=textcolor, alpha=0.88)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("",
                xy=(x1, y1), xycoords="axes fraction",
                xytext=(x0, y0), textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#555555",
                    lw=1.6,
                    mutation_scale=14,
                ))


def make_architecture():
    fig, ax = plt.subplots(figsize=(14, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(0.5, 0.95, "Credit Risk Intelligence Architecture",
            ha="center", va="top", fontsize=20, fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89,
            "Four modality encoders  →  late fusion  →  explainable credit decision",
            ha="center", va="top", fontsize=11, color="#444444")

    # Column x-positions and widths
    src_x, src_w = 0.04, 0.17
    enc_x, enc_w = 0.27, 0.17
    fus_x, fus_w = 0.53, 0.16
    out_x, out_w = 0.77, 0.18

    # Row y-positions (centre of each row)
    ys   = [0.68, 0.52, 0.36, 0.20]
    box_h = 0.11

    src_labels = [
        ("Tabular Data",   "GiveMeSomeCredit\ncredit features"),
        ("Time Series",    "12-month\npayment sequence"),
        ("Graph Data",     "Borrower similarity\nnetwork"),
        ("Loan Text",      "Synthetic loan\ndescriptions"),
    ]
    enc_labels = [
        ("XGBoost",        "Leaf Embedding\n32-dim"),
        ("LSTM",           "Sequence Encoder\n32-dim"),
        ("GraphSAGE",      "Node Encoder\n32-dim"),
        ("sentence-BERT",  "Frozen MiniLM\n32-dim"),
    ]

    # Source boxes (indigo)
    for (title, sub), y in zip(src_labels, ys):
        _box(ax, src_x, y - box_h / 2, src_w, box_h, title, sub,
             facecolor="#5B5EA6")

    # Encoder boxes (teal)
    for (title, sub), y in zip(enc_labels, ys):
        _box(ax, enc_x, y - box_h / 2, enc_w, box_h, title, sub,
             facecolor="#2E8B6E")

    # Arrows: source -> encoder
    for y in ys:
        _arrow(ax, src_x + src_w, y, enc_x, y)

    # Arrows: encoder -> fusion (fan in)
    fus_cy = 0.44
    for y in ys:
        _arrow(ax, enc_x + enc_w, y, fus_x, fus_cy)

    # Fusion MLP box (orange)
    fus_h = 0.30
    _box(ax, fus_x, fus_cy - fus_h / 2, fus_w, fus_h,
         "Late Fusion MLP",
         "128-dim concat\n→ risk logit",
         facecolor="#E07B39", fontsize=12)

    # Output boxes
    out_configs = [
        (0.70, "Risk Score",    "P(default)",          "#C0392B"),
        (0.50, "SHAP + Report", "Explainability\n+ GenAI", "#555555"),
        (0.28, "Streamlit",     "Dashboard",           "#2980B9"),
    ]
    for y, title, sub, color in out_configs:
        _box(ax, out_x, y - box_h / 2, out_w, box_h, title, sub,
             facecolor=color)
        _arrow(ax, fus_x + fus_w, fus_cy, out_x, y)

    # Footer
    ax.text(0.5, 0.03,
            "Modular design: each encoder outputs a fixed 32-dimensional borrower representation",
            ha="center", va="bottom", fontsize=9, color="#333333")

    out_path = FIGURES_DIR / "architecture_overview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] architecture_overview.png -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Risk scoring table
# ---------------------------------------------------------------------------

def make_risk_table():
    # Set CJK font so risk_level cells render correctly
    plt.rcParams["font.family"] = CJK_FONT

    rows = [
        ("VAL-000", "50.7%", "中風險", "RevolvingUtilization..."),
        ("VAL-001", "48.2%", "中風險", "DebtRatio"),
        ("VAL-002", "61.4%", "高風險", "NumberOfTimes90DaysLate"),
    ]
    col_headers = ["borrower_id", "risk_score", "risk_level", "top_shap_feature"]

    # Colours: header dark navy, rows alternating white/light, risk_level tinted
    HEADER_BG  = "#1a2744"
    HEADER_FG  = "white"
    CELL_BG    = "white"
    RISK_BG    = "#fde8e8"   # light red tint for risk_level cell

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(0.5, 0.92, "Demo Risk Scoring Results",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="#1a2744",
            fontfamily="DejaVu Sans")
    ax.text(0.5, 0.80, "Fusion model output with strongest tabular SHAP driver",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11, color="#666666",
            fontfamily="DejaVu Sans")

    # Table geometry (axes fraction)
    tbl_left, tbl_right = 0.05, 0.95
    tbl_top,  tbl_bot   = 0.70, 0.15
    col_rights = [0.22, 0.38, 0.55, 1.0]   # cumulative widths as fractions of table width
    col_xs = [tbl_left + (tbl_right - tbl_left) * x for x in [0] + col_rights]
    row_ys = np.linspace(tbl_top, tbl_bot, len(rows) + 2)  # header + 3 data rows + bottom

    def cell(ax, x0, x1, y0, y1, text, bg, fg="black", fs=11, bold=False, font=None):
        rect = mpatches.FancyBboxPatch(
            (x0, y1), x1 - x0, y0 - y1,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor="#cccccc", linewidth=0.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        kw = dict(ha="center", va="center",
                  transform=ax.transAxes,
                  fontsize=fs, color=fg,
                  fontweight="bold" if bold else "normal")
        if font:
            kw["fontfamily"] = font
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, **kw)

    # Header row
    for i, (x0, x1, hdr) in enumerate(zip(col_xs[:-1], col_xs[1:], col_headers)):
        cell(ax, x0, x1, row_ys[0], row_ys[1], hdr,
             bg=HEADER_BG, fg=HEADER_FG, bold=True, font="DejaVu Sans")

    # Data rows
    for r, (bid, score, level, feature) in enumerate(rows):
        y0, y1 = row_ys[r + 1], row_ys[r + 2]
        values = [bid, score, level, feature]
        for c, (x0, x1, val) in enumerate(zip(col_xs[:-1], col_xs[1:], values)):
            bg = RISK_BG if c == 2 else CELL_BG
            # Use CJK font only for the risk_level column
            font = CJK_FONT if c == 2 else "DejaVu Sans"
            cell(ax, x0, x1, y0, y1, val, bg=bg, font=font)

    # Footer
    ax.text(0.5, 0.06, "White-background slide visual  1200x800 PNG",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=8, color="#aaaaaa", fontfamily="DejaVu Sans")

    out_path = FIGURES_DIR / "risk_scoring_table.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Reset font so subsequent plots aren't affected
    plt.rcParams["font.family"] = "DejaVu Sans"
    print(f"[OK] risk_scoring_table.png -> {out_path}")


# ---------------------------------------------------------------------------
# 3. SHAP summary (Chinese)
# ---------------------------------------------------------------------------

def make_shap_zh():
    import shap
    import pandas as pd

    # Use test split (same data the original plot used)
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        test_path = PROCESSED_DIR / "val.parquet"   # fallback
    df = pd.read_parquet(test_path)
    X = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    with (PROCESSED_DIR / "xgb_baseline.pkl").open("rb") as fh:
        model = pickle.load(fh)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Build Chinese feature name list in the same column order
    zh_names = [FEATURE_ZH.get(c, c) for c in X.columns]

    # JhengHei covers CJK; DejaVu Sans fallback supplies the U+2212 minus sign
    # used on the x-axis tick labels, which JhengHei lacks.
    plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=zh_names,
        show=False,
        plot_size=None,   # use our fig/ax sizing
    )

    # Override axis labels to Chinese
    cur_ax = plt.gca()
    cur_ax.set_xlabel("SHAP 值（對模型輸出的影響）", fontsize=11, fontfamily=CJK_FONT)
    cur_ax.tick_params(axis="y", labelsize=10)

    # Colorbar label
    cbar = fig.axes[-1] if len(fig.axes) > 1 else None
    if cbar is not None:
        cbar.set_ylabel("特徵值", fontsize=10, fontfamily=CJK_FONT)
        cbar.set_yticks([0, 1])
        cbar.set_yticklabels(["低", "高"], fontsize=9, fontfamily=CJK_FONT)

    plt.tight_layout()
    out_path = FIGURES_DIR / "shap_summary_day1.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close("all")
    plt.rcParams["font.family"] = "DejaVu Sans"
    print(f"[OK] shap_summary_day1.png -> {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_architecture()
    make_risk_table()
    make_shap_zh()
    print("\nAll figures regenerated.")
