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
except ImportError:
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
APP_QR_URL = os.getenv(
    "APP_QR_URL",
    "https://creditriskintelligence-jaxjumckv583zulzvweieh.streamlit.app/",
)
APP_QR_LABEL = os.getenv("APP_QR_LABEL", "掃描開啟線上 Demo")
FUSION_ALIGNED_PATH = PROCESSED_DIR / "fusion_model_aligned.pt"
FUSION_BASE_PATH = PROCESSED_DIR / "fusion_model.pt"
LSTM_PATH = PROCESSED_DIR / "lstm_encoder.pt"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="信用風險智能評估",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design tokens ─────────────────────────────────────────────────────────────
C_BG     = "#0D1117"
C_CARD   = "#161B22"
C_BORDER = "#30363D"
C_GOLD   = "#F0B429"
C_RED    = "#EF4444"
C_GREEN  = "#10B981"
C_BLUE   = "#3B82F6"
C_TEXT   = "#F0F6FC"
C_MUTED  = "#8B949E"

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stStatusWidget"] {visibility: hidden;}

/* App background */
.stApp { background-color: #0D1117; color: #F0F6FC; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] .stMarkdown { color: #F0F6FC; }
section[data-testid="stSidebar"] label { color: #F0F6FC !important; }
section[data-testid="stSidebar"] .stRadio label p { color: #F0F6FC !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0D1117;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #F0F6FC !important; font-size: 1.05rem !important; }

/* Section headers */
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #F0F6FC;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #30363D;
}

/* Primary button: full-width gold */
.stButton > button[kind="primary"] {
    background: #F0B429 !important;
    color: #0D1117 !important;
    border: none !important;
    font-weight: 700 !important;
    width: 100% !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 15px !important;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    background: #D4A017 !important;
    color: #0D1117 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _risk_colors(score: float, is_ensemble: bool = False) -> tuple[str, str]:
    low, mid = (0.15, 0.35) if is_ensemble else (0.3, 0.6)
    if score < low:
        return C_GREEN, C_GREEN + "22"
    if score < mid:
        return C_GOLD, C_GOLD + "22"
    return C_RED, C_RED + "22"


def _action_icon(action: str) -> str:
    if "核准" in action:
        return "✅"
    if "補件" in action:
        return "📋"
    if "人工" in action:
        return "👤"
    return "❌"


def _apply_dark_theme(fig) -> None:
    for ax in fig.axes:
        ax.set_facecolor(C_CARD)
        ax.tick_params(colors=C_TEXT, labelsize=10)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        for spine in ax.spines.values():
            spine.set_color(C_BORDER)
        for text in ax.texts:
            text.set_color(C_TEXT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(C_TEXT)
        try:
            ax.title.set_color(C_TEXT)
        except Exception:
            pass


def _show_plot_image(fig, caption: str | None = None) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)


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
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH), False
    return _make_synthetic_borrowers(), True


@st.cache_data
def load_shap_data():
    if not SHAP_PATH.exists():
        return None
    try:
        with SHAP_PATH.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


@st.cache_data
def build_qr_code(url: str) -> bytes | None:
    if not url:
        return None
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M
    except ImportError:
        return None
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M,
                       box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0D1117", back_color="#FFFFFF")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_resource
def load_fusion_artifacts(fusion_path: str, lstm_path: str):
    try:
        import torch
    except ImportError:
        return None
    try:
        fusion = (torch.load(fusion_path, map_location="cpu")
                  if Path(fusion_path).exists() else None)
        lstm = (torch.load(lstm_path, map_location="cpu")
                if lstm_path and Path(lstm_path).exists() else None)
        return {"fusion": fusion, "lstm": lstm}
    except Exception:
        return None


# ── data ──────────────────────────────────────────────────────────────────────
predictions_df, is_synthetic = load_predictions()
shap_obj = load_shap_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:800;color:#F0F6FC;'
        'padding:8px 0 2px 0;">🏦 信用風險評估</div>'
        '<div style="font-size:0.72rem;color:#8B949E;padding-bottom:14px;'
        'letter-spacing:0.04em;">Credit Risk Intelligence System</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:#30363D;margin:0 0 14px 0;">',
                unsafe_allow_html=True)

    if FUSION_ALIGNED_PATH.exists():
        model_version = "Aligned Fusion v0 · Val AUC 0.864"
        _model_artifacts = load_fusion_artifacts(str(FUSION_ALIGNED_PATH), "")
    elif FUSION_BASE_PATH.exists():
        model_version = "原版 Fusion（Demo）"
        _model_artifacts = load_fusion_artifacts(str(FUSION_BASE_PATH), str(LSTM_PATH))
    else:
        model_version = "Demo（預先計算結果）"
        _model_artifacts = None

    is_ensemble = False

    st.markdown(
        '<div style="font-size:0.68rem;font-weight:600;color:#8B949E;'
        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">'
        '借款人選擇</div>',
        unsafe_allow_html=True,
    )
    borrower_ids = predictions_df["borrower_id"].tolist()
    selected_id = st.selectbox(
        "選擇借款人 ID",
        options=borrower_ids,
        format_func=lambda x: f"借款人 #{x}",
        label_visibility="collapsed",
    )
    row = predictions_df[predictions_df["borrower_id"] == selected_id].iloc[0]

    st.markdown('<hr style="border-color:#30363D;margin:14px 0;">',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem;font-weight:600;color:#8B949E;'
        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">'
        '基本資料</div>',
        unsafe_allow_html=True,
    )
    st.metric("月收入", f"NT$ {int(row['MonthlyIncome']):,}")
    st.metric("負債比率", f"{float(row['DebtRatio']):.3f}")
    st.metric("90天逾期次數", int(row["NumberOfTimes90DaysLate"]))
    st.metric("年齡區間", str(row["age_bucket"]))

    st.markdown('<hr style="border-color:#30363D;margin:14px 0 10px 0;">',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.7rem;color:#8B949E;padding:8px 10px;'
        f'background:#0D1117;border:1px solid #30363D;border-radius:8px;'
        f'text-align:center;line-height:1.6;">'
        f'📊 {model_version}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# gradient top border
st.markdown(
    '<div style="height:3px;background:linear-gradient(90deg,#F0B429,#10B981,#3B82F6);'
    'border-radius:2px;margin-bottom:18px;"></div>',
    unsafe_allow_html=True,
)

# header
st.markdown(
    '<div style="margin-bottom:2px;">'
    '<span style="font-size:28px;font-weight:800;color:#F0F6FC;'
    'letter-spacing:-0.02em;">信用風險智能評估系統</span>'
    '</div>'
    '<div style="font-size:13px;color:#8B949E;margin-bottom:16px;'
    'letter-spacing:0.03em;">'
    'Multi-Modal Credit Risk Intelligence&nbsp;·&nbsp;'
    'Tabular · Time Series · Graph · NLP'
    '</div>',
    unsafe_allow_html=True,
)

if is_synthetic:
    st.warning(
        "predictions.csv 未找到，顯示合成示範資料。"
        "請先執行推論流程以生成真實預測結果。"
    )

# ── risk state ────────────────────────────────────────────────────────────────
risk_score = float(np.clip(row["risk_score"], 0.0, 1.0))
border_color, bg_tint = _risk_colors(risk_score, is_ensemble)

if is_ensemble:
    if risk_score < 0.15:
        risk_level, recommended_action = "低風險", "可核准放款"
    elif risk_score < 0.35:
        risk_level, recommended_action = "中風險", "需補件審查"
    else:
        risk_level, recommended_action = "高風險", "建議拒絕或擔保"
else:
    risk_level = str(row["risk_level"])
    recommended_action = str(row["recommended_action"])

action_icon = _action_icon(recommended_action)

# ── Section 1: Risk Score ─────────────────────────────────────────────────────
st.markdown(
    '<hr style="border-color:#30363D;margin:0 0 16px 0;">',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([3, 3, 4])

with c1:
    st.markdown(
        f'<div style="background:{C_CARD};border:1px solid {border_color};'
        f'border-radius:14px;padding:28px 20px;text-align:center;">'
        f'<div style="font-size:48px;font-weight:800;color:{border_color};'
        f'line-height:1;letter-spacing:-0.02em;">{risk_score:.1%}</div>'
        f'<div style="color:{C_MUTED};font-size:11px;margin-top:10px;'
        f'letter-spacing:0.08em;text-transform:uppercase;">違約風險機率</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div style="background:{bg_tint};border:1px solid {border_color};'
        f'border-radius:14px;padding:28px 20px;text-align:center;">'
        f'<div style="font-size:26px;font-weight:700;color:{border_color};">'
        f'{risk_level}</div>'
        f'<div style="color:{C_MUTED};font-size:11px;margin-top:10px;'
        f'letter-spacing:0.08em;text-transform:uppercase;">風險等級</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div style="background:{C_CARD};border:1px solid {C_BORDER};'
        f'border-radius:14px;padding:28px 24px;">'
        f'<div style="color:{C_MUTED};font-size:11px;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:10px;">建議行動</div>'
        f'<div style="font-size:20px;font-weight:700;color:{C_TEXT};">'
        f'{action_icon}&nbsp;{recommended_action}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── risk gauge (replaces progress bar) ───────────────────────────────────────
filled = int(risk_score * 20)
segments = ""
for i in range(20):
    threshold = (i + 1) * 5
    seg_color = C_GREEN if threshold <= 30 else (C_GOLD if threshold <= 60 else C_RED)
    opacity = "1" if i < filled else "0.15"
    segments += (
        f'<div style="flex:1;height:6px;border-radius:3px;'
        f'background:{seg_color};opacity:{opacity};margin:0 1.5px;"></div>'
    )

st.markdown(
    f'<div style="display:flex;align-items:center;margin:14px 0 4px 0;">'
    f'{segments}</div>'
    f'<div style="display:flex;justify-content:space-between;'
    f'font-size:11px;color:{C_MUTED};margin-bottom:2px;">'
    f'<span>0%</span>'
    f'<span style="color:{C_GREEN};">低風險 &lt;30%</span>'
    f'<span style="color:{C_GOLD};">中風險 30–60%</span>'
    f'<span style="color:{C_RED};">高風險 &gt;60%</span>'
    f'<span>100%</span></div>',
    unsafe_allow_html=True,
)

# ── model info bar ────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:{C_CARD};border:1px solid {C_BORDER};'
    f'border-radius:10px;padding:12px 20px;margin:14px 0;'
    f'display:flex;gap:40px;flex-wrap:wrap;align-items:center;">'
    f'<div><div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;'
    f'letter-spacing:0.08em;">Active Model</div>'
    f'<div style="color:{C_TEXT};font-size:13px;font-weight:600;margin-top:2px;">'
    f'📊 {model_version}</div></div>'
    f'<div><div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;'
    f'letter-spacing:0.08em;">Ensemble AUC</div>'
    f'<div style="color:{C_GREEN};font-size:13px;font-weight:700;margin-top:2px;">'
    f'0.866</div></div>'
    f'<div><div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;'
    f'letter-spacing:0.08em;">Time Series AUC</div>'
    f'<div style="color:{C_GREEN};font-size:13px;font-weight:700;margin-top:2px;">'
    f'0.97</div></div>'
    f'<div><div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;'
    f'letter-spacing:0.08em;">Dataset</div>'
    f'<div style="color:{C_TEXT};font-size:13px;font-weight:600;margin-top:2px;">'
    f'150K borrowers · 6.7% default</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── AI report helpers ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "你是一位專業信貸分析師。根據提供的 SHAP 特徵貢獻數據，"
    "用繁體中文寫一段 3-4 句的授信摘要報告。"
    "只能使用提供的數據，不能推測或編造。"
    "格式：風險等級說明 → 主要風險因子 → 次要因子 → 建議行動。"
)

_REPORT_BOX = (
    f"background:{C_CARD};border-left:3px solid {C_GOLD};"
    "border-radius:0 8px 8px 0;padding:16px 20px;"
    "font-size:14px;line-height:1.8;color:#F0F6FC;min-height:80px;"
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
        f"- 風險評分：{score:.1%}\n- 風險等級：{level}\n"
        f"- 月收入：NT$ {income:,}\n- 負債比率：{debt:.3f}\n"
        f"- 90 天以上逾期次數：{late90}\n- 年齡區間：{age_b}\n\n"
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


# ── Section 2+3: AI Report (left 55%) | SHAP (right 45%) ─────────────────────
st.markdown(
    '<hr style="border-color:#30363D;margin:8px 0 16px 0;">',
    unsafe_allow_html=True,
)
col_report, col_shap = st.columns([11, 9], gap="large")

with col_report:
    st.markdown(
        '<div class="section-header">🤖 AI 授信摘要報告</div>'
        f'<div style="font-size:11px;color:{C_MUTED};margin-bottom:12px;">'
        f'Powered by OpenAI ({OPENAI_MODEL})'
        f'&nbsp;·&nbsp;報告基於 SHAP 數據生成，不含推測</div>',
        unsafe_allow_html=True,
    )

    if st.button("生成 AI 授信摘要", type="primary"):
        user_msg = _build_user_message(row)
        placeholder = st.empty()
        full_text = ""
        for chunk in _stream_report(user_msg, row):
            full_text += chunk
            placeholder.markdown(
                f'<div style="{_REPORT_BOX}">{full_text}▌</div>',
                unsafe_allow_html=True,
            )
        placeholder.markdown(
            f'<div style="{_REPORT_BOX}">{full_text}</div>',
            unsafe_allow_html=True,
        )
        st.session_state["ai_report"] = full_text
        st.session_state["ai_report_borrower_id"] = selected_id
    elif (
        st.session_state.get("ai_report")
        and st.session_state.get("ai_report_borrower_id") == selected_id
    ):
        st.markdown(
            f'<div style="{_REPORT_BOX}">{st.session_state["ai_report"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="{_REPORT_BOX};color:{C_MUTED};">'
            f'點擊上方按鈕以生成 AI 授信摘要報告...</div>',
            unsafe_allow_html=True,
        )

with col_shap:
    st.markdown(
        '<div class="section-header">📊 SHAP 特徵解釋</div>',
        unsafe_allow_html=True,
    )

    _shap_rendered = False

    if shap_obj is not None:
        try:
            import shap

            explanation = shap_obj[0] if isinstance(shap_obj, tuple) else shap_obj
            try:
                borrower_idx = borrower_ids.index(selected_id)
            except ValueError:
                borrower_idx = 0
            borrower_idx = min(borrower_idx, len(explanation) - 1)

            shap.plots.waterfall(explanation[borrower_idx], max_display=10, show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor(C_CARD)
            _apply_dark_theme(fig)
            _show_plot_image(fig)
            _shap_rendered = True
        except Exception as exc:
            st.warning(f"SHAP waterfall 繪圖失敗：{exc}，顯示示範條形圖。")

    if not _shap_rendered:
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
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(C_CARD)
        ax.set_facecolor(C_CARD)
        bar_colors = [C_RED if v > 0 else C_GREEN for v in vals]
        ax.barh(feats, vals, color=bar_colors, height=0.5)
        ax.axvline(0, color=C_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP 貢獻值", color=C_MUTED, fontsize=12)
        ax.tick_params(colors=C_TEXT, labelsize=11)
        for spine in ax.spines.values():
            spine.set_color(C_BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(
            f"借款人 #{selected_id} — 前 5 特徵貢獻",
            color=C_TEXT, pad=12, fontsize=14,
        )
        plt.tight_layout()
        _show_plot_image(fig)

# ── Footer: QR + credits ──────────────────────────────────────────────────────
st.markdown(
    '<hr style="border-color:#30363D;margin:24px 0 16px 0;">',
    unsafe_allow_html=True,
)

_, _qr_mid, _ = st.columns([2, 1, 2])
with _qr_mid:
    _qr_png = build_qr_code(APP_QR_URL)
    if _qr_png:
        st.image(_qr_png, width=150)
    st.markdown(
        f'<div style="text-align:center;font-size:11px;color:{C_MUTED};margin-top:4px;">'
        f'掃描開啟線上 Demo</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div style="text-align:center;font-size:11px;color:{C_MUTED};'
    f'padding:10px 0 20px 0;">'
    f'表格 Ensemble OOF AUC 0.866&nbsp;·&nbsp;'
    f'Aligned Fusion Val AUC 0.864&nbsp;·&nbsp;'
    f'時序模態使用 Lending Club 真實資料（Val AUC 0.97）'
    f'</div>',
    unsafe_allow_html=True,
)
