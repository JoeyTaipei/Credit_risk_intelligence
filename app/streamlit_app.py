from __future__ import annotations

import os
import pickle
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_CJK_CANDIDATES = ["Noto Sans TC", "Noto Sans CJK TC", "Noto Sans HK",
                   "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
for _f in _CJK_CANDIDATES:
    if any(_f.lower() in _font.name.lower() for _font in _fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"
SHAP_PATH = PROCESSED_DIR / "shap_values.pkl"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="信用風險智能評估",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    section[data-testid="stSidebar"] .stMarkdown { color: #E6EDF3; }
    [data-testid="stMetric"] { background: #161B22; border-radius: 8px; padding: 12px; }
    [data-testid="stMetricLabel"] { color: #8B949E !important; }
    [data-testid="stMetricValue"] { color: #E6EDF3 !important; }
    .stProgress > div > div { background-color: #3FB950; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── synthetic fallback ────────────────────────────────────────────────────────
_FEATURE_POOL = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
]


def _make_synthetic_borrowers() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 10
    scores = rng.uniform(0.1, 0.9, n).round(4)
    return pd.DataFrame({
        "borrower_id": list(range(n)),
        "risk_score": scores,
        "risk_level": [
            "低風險" if s < 0.3 else ("中風險" if s < 0.6 else "高風險") for s in scores
        ],
        "recommended_action": [
            "可核准放款" if s < 0.3 else ("需補件審查" if s < 0.6 else "建議拒絕或擔保")
            for s in scores
        ],
        "top_shap_feature": rng.choice(_FEATURE_POOL, n),
        "shap_value_top1": rng.uniform(0.05, 0.4, n).round(4),
        "MonthlyIncome": rng.integers(30_000, 150_001, n),
        "DebtRatio": rng.uniform(0.1, 0.9, n).round(3),
        "NumberOfTimes90DaysLate": rng.integers(0, 6, n),
        "age_bucket": rng.choice(["20-30", "31-40", "41-50", "51-60", "61+"], n),
    })


# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_synthetic). Cached so CSV is read once per session."""
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH), False
    return _make_synthetic_borrowers(), True


@st.cache_data
def load_shap_data():
    """Returns a shap.Explanation (or raw array), or None if file missing or incompatible."""
    if not SHAP_PATH.exists():
        return None
    try:
        with SHAP_PATH.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────
def _risk_hex(score: float) -> str:
    if score < 0.3:
        return "#3FB950"   # green
    return "#D29922" if score < 0.6 else "#F85149"   # yellow / red


def _show_plot_image(fig, caption: str | None = None) -> None:
    """Render matplotlib output as a PNG image so Streamlit always receives an image."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    st.image(buffer, caption=caption, use_container_width=True)
    plt.close(fig)


# ── data ─────────────────────────────────────────────────────────────────────
predictions_df, is_synthetic = load_predictions()
shap_obj = load_shap_data()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 借款人選擇")
    borrower_ids = predictions_df["borrower_id"].tolist()
    selected_id = st.selectbox(
        "選擇借款人 ID",
        options=borrower_ids,
        format_func=lambda x: f"借款人 #{x}",
    )
    row = predictions_df[predictions_df["borrower_id"] == selected_id].iloc[0]

    st.markdown("---")
    st.markdown("### 基本資料")
    st.metric("月收入 (MonthlyIncome)", f"NT$ {int(row['MonthlyIncome']):,}")
    st.metric("負債比率 (DebtRatio)", f"{float(row['DebtRatio']):.3f}")
    st.metric("90 天以上逾期次數", int(row["NumberOfTimes90DaysLate"]))
    st.metric("年齡區間 (age_bucket)", str(row["age_bucket"]))

# ── page header ───────────────────────────────────────────────────────────────
st.title("信用風險智能評估系統")
st.caption("Multi-Modal AI Pipeline: Tabular + Time Series + Graph + NLP")

if is_synthetic:
    st.warning(
        "predictions.csv 未找到，顯示合成示範資料。"
        "請先執行推論流程以生成真實預測結果。"
    )

risk_score = float(np.clip(row["risk_score"], 0.0, 1.0))
risk_level = str(row["risk_level"])
recommended_action = str(row["recommended_action"])
color = _risk_hex(risk_score)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Risk Score
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 風險評分")

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.markdown(
        f'<div style="background:#161B22;border:2px solid {color};border-radius:12px;'
        f'padding:24px;text-align:center;">'
        f'<div style="font-size:3rem;font-weight:700;color:{color};">{risk_score:.1%}</div>'
        f'<div style="color:#8B949E;font-size:0.85rem;margin-top:6px;">違約風險機率</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div style="background:#161B22;border:2px solid {color};border-radius:12px;'
        f'padding:24px;text-align:center;">'
        f'<div style="font-size:2rem;font-weight:700;color:{color};">{risk_level}</div>'
        f'<div style="color:#8B949E;font-size:0.85rem;margin-top:6px;">風險等級</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div style="background:#161B22;border:1px solid #30363D;border-radius:12px;'
        f'padding:24px;">'
        f'<div style="color:#8B949E;font-size:0.85rem;margin-bottom:8px;">建議行動</div>'
        f'<div style="font-size:1.2rem;font-weight:600;color:#E6EDF3;">{recommended_action}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.progress(risk_score)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — SHAP Explanation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## SHAP 特徵解釋")

_shap_rendered = False

if shap_obj is not None:
    try:
        import shap

        # pkl may be a bare Explanation or wrapped in a tuple
        explanation = shap_obj[0] if isinstance(shap_obj, tuple) else shap_obj

        # map selected borrower to its position in the shap array
        try:
            borrower_idx = borrower_ids.index(selected_id)
        except ValueError:
            borrower_idx = 0
        borrower_idx = min(borrower_idx, len(explanation) - 1)

        shap.plots.waterfall(explanation[borrower_idx], max_display=10, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0D1117")
        _show_plot_image(fig, caption=f"借款人 #{selected_id} SHAP waterfall")
        _shap_rendered = True
    except Exception as exc:
        st.warning(f"SHAP waterfall 繪圖失敗：{exc}，顯示示範條形圖。")

if not _shap_rendered:
    if shap_obj is None:
        st.warning("shap_values.pkl 未找到，顯示示範特徵貢獻圖。")

    # seed on borrower so each selection looks distinct
    seed = abs(hash(str(selected_id))) % (2 ** 32)
    rng = np.random.default_rng(seed)
    feats = [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfTime30-59DaysPastDueNotWorse",
    ]
    vals = rng.uniform(-0.3, 0.4, 5)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#161B22")
    ax.set_facecolor("#161B22")
    bar_colors = ["#F85149" if v > 0 else "#3FB950" for v in vals]
    ax.barh(feats, vals, color=bar_colors, height=0.5)
    ax.axvline(0, color="#8B949E", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP 貢獻值", color="#8B949E", fontsize=11)
    ax.tick_params(colors="#E6EDF3", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#30363D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"借款人 #{selected_id} — 前 5 特徵貢獻（示範）",
        color="#E6EDF3",
        pad=12,
        fontsize=13,
    )
    plt.tight_layout()
    _show_plot_image(fig, caption=f"借款人 #{selected_id} 示範 SHAP 特徵貢獻")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — AI Credit Report
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## AI 授信摘要報告")
st.caption(f"Powered by OpenAI ({OPENAI_MODEL}) — 報告基於 SHAP 數據生成，不含推測")

_SYSTEM_PROMPT = (
    "你是一位專業信貸分析師。根據提供的 SHAP 特徵貢獻數據，"
    "用繁體中文寫一段 3-4 句的授信摘要報告。"
    "只能使用提供的數據，不能推測或編造。"
    "格式：風險等級說明 → 主要風險因子 → 次要因子 → 建議行動。"
)


def _build_user_message(r: pd.Series) -> str:
    top_feat = str(r.get("top_shap_feature", "N/A"))
    shap1 = float(r.get("shap_value_top1", 0.0))
    income = int(r.get("MonthlyIncome", 0))
    debt = float(r.get("DebtRatio", 0.0))
    late90 = int(r.get("NumberOfTimes90DaysLate", 0))
    age_b = str(r.get("age_bucket", "N/A"))
    score = float(r.get("risk_score", 0.0))
    level = str(r.get("risk_level", "N/A"))

    shap2 = round(max(0.01, debt * 0.3), 4)
    shap3 = round(min(0.5, late90 * 0.08), 4)

    return (
        f"借款人資料：\n"
        f"- 風險評分：{score:.1%}\n"
        f"- 風險等級：{level}\n"
        f"- 月收入：NT$ {income:,}\n"
        f"- 負債比率：{debt:.3f}\n"
        f"- 90 天以上逾期次數：{late90}\n"
        f"- 年齡區間：{age_b}\n\n"
        f"前 3 SHAP 特徵貢獻：\n"
        f"1. {top_feat}（SHAP 值：{shap1:.4f}）\n"
        f"2. DebtRatio（SHAP 值：{shap2}）\n"
        f"3. NumberOfTimes90DaysLate（SHAP 值：{shap3}）\n"
    )


def _format_local_report(r: pd.Series) -> str:
    top_feat = str(r.get("top_shap_feature", "N/A"))
    shap1 = float(r.get("shap_value_top1", 0.0))
    income = int(r.get("MonthlyIncome", 0))
    debt = float(r.get("DebtRatio", 0.0))
    late90 = int(r.get("NumberOfTimes90DaysLate", 0))
    score = float(r.get("risk_score", 0.0))
    level = str(r.get("risk_level", "N/A"))
    action = str(r.get("recommended_action", "請信貸人員複核後決定"))

    return (
        f"本筆借款人的違約風險評分為 **{score:.1%}**，目前歸類為 **{level}**。"
        f"主要風險訊號為 **{top_feat}**（SHAP 值：`{shap1:.4f}`）。"
        f"次要審查重點包含負債比率 `{debt:.3f}`、90 天以上逾期次數 `{late90}` 次，"
        f"以及月收入 `NT$ {income:,}` 的償債支撐。建議行動：**{action}**。"
    )


def _stream_report(user_msg: str, r: pd.Series):
    """Stream OpenAI's response token by token; fall back to local rule report."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield _format_local_report(r)
        return

    try:
        from openai import OpenAI
    except ImportError:
        yield _format_local_report(r)
        return

    try:
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        yield _format_local_report(r)


if st.button("生成 AI 授信摘要", type="primary"):
    user_msg = _build_user_message(row)
    st.session_state["ai_report_borrower_id"] = selected_id
    st.session_state["ai_report"] = st.write_stream(_stream_report(user_msg, row))
elif (
    st.session_state.get("ai_report")
    and st.session_state.get("ai_report_borrower_id") == selected_id
):
    st.markdown(st.session_state["ai_report"])
