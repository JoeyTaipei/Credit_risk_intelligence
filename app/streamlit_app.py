from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import build_borrower_graph
from src.data.preprocess import clean_tabular, engineer_tabular_features
from src.inference.predict import predict_single_borrower
from src.models.fusion import LateFusionClassifier
from src.models.gnn_encoder import GraphSAGEEncoder
from src.models.lstm_encoder import LSTMEncoder
from src.models.text_encoder import TextEncoder
from src.utils.openai_report import generate_credit_report

st.set_page_config(layout="wide", page_title="信用風險智能評估")
st.title("信用風險智能評估系統")
st.caption("Multi-Modal AI Pipeline: Tabular + Time Series + Graph + NLP")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DEMO_INPUT_PATH = PROJECT_ROOT / "data" / "demo_input.csv"
LEGACY_DEMO_INPUT_PATH = PROJECT_ROOT / "demo_input.csv"
DEFAULT_DESCRIPTION = "請輸入借款用途、資金需求原因與還款計畫。"


def _required_model_paths() -> dict[str, Path]:
    return {
        "xgb": PROCESSED_DIR / "xgb_baseline.pkl",
        "lstm": PROCESSED_DIR / "lstm_encoder.pt",
        "gnn": PROCESSED_DIR / "gnn_encoder.pt",
        "fusion": PROCESSED_DIR / "fusion_model.pt",
        "train": PROCESSED_DIR / "train.parquet",
    }


@st.cache_resource
def load_models() -> dict:
    paths = _required_model_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    with paths["xgb"].open("rb") as file:
        xgb_model = pickle.load(file)

    lstm_encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    lstm_encoder.load_state_dict(torch.load(paths["lstm"], map_location="cpu"))
    lstm_encoder.eval()

    gnn_encoder = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    gnn_encoder.load_state_dict(torch.load(paths["gnn"], map_location="cpu"))
    gnn_encoder.eval()

    fusion_model = LateFusionClassifier()
    fusion_model.load_state_dict(torch.load(paths["fusion"], map_location="cpu"))
    fusion_model.eval()

    torch.manual_seed(42)
    text_encoder = TextEncoder(freeze=True)
    text_encoder.eval()

    train_df = pd.read_parquet(paths["train"])
    edge_index, node_features = build_borrower_graph(train_df)
    with torch.no_grad():
        node_embedding = gnn_encoder(node_features, edge_index).mean(dim=0)

    return {
        "xgb": xgb_model,
        "lstm_encoder": lstm_encoder,
        "gnn_encoder": gnn_encoder,
        "text_encoder": text_encoder,
        "fusion_model": fusion_model,
        "node_embedding": node_embedding,
    }


def _prepare_tabular(df: pd.DataFrame) -> pd.DataFrame:
    if "income_per_dependent" in df.columns and "total_past_due" in df.columns:
        return df.copy()
    return engineer_tabular_features(clean_tabular(df))


def _risk_color(level: str) -> str:
    if level == "低風險":
        return "green"
    if level == "中風險":
        return "orange"
    return "red"


def _embedding_norms(embeddings: dict) -> pd.DataFrame:
    labels = {
        "tabular": "表格",
        "lstm": "時序",
        "gnn": "圖網路",
        "text": "文字",
    }
    rows = []
    for key, value in embeddings.items():
        tensor = torch.tensor(value, dtype=torch.float)
        rows.append({"模態": labels.get(key, key), "模態信號強度": float(torch.linalg.norm(tensor))})
    return pd.DataFrame(rows).set_index("模態")


def _plot_embedding_norms(norms: pd.DataFrame) -> alt.Chart:
    chart_df = norms.reset_index()
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "模態:N",
                sort=None,
                axis=alt.Axis(title=None, labelAngle=0, labelFontSize=17, labelColor="#111827"),
            ),
            y=alt.Y(
                "模態信號強度:Q",
                axis=alt.Axis(title="模態信號強度", titleFontSize=14, labelFontSize=12),
            ),
            color=alt.Color("模態:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["模態:N", alt.Tooltip("模態信號強度:Q", format=".3f")],
        )
        .properties(height=320)
    )


def _feature_name_zh(name: str) -> str:
    mapping = {
        "RevolvingUtilizationOfUnsecuredLines": "循環信用使用率",
        "age": "年齡",
        "NumberOfTime30-59DaysPastDueNotWorse": "30-59 天逾期次數",
        "DebtRatio": "負債比率",
        "MonthlyIncome": "月收入",
        "NumberOfOpenCreditLinesAndLoans": "開放信用額度與貸款數",
        "NumberOfTimes90DaysLate": "90 天以上逾期次數",
        "NumberRealEstateLoansOrLines": "不動產貸款或額度數",
        "NumberOfTime60-89DaysPastDueNotWorse": "60-89 天逾期次數",
        "NumberOfDependents": "扶養人數",
        "income_per_dependent": "每位家庭成員收入",
        "total_past_due": "總逾期次數",
        "has_any_delinquency": "是否有逾期紀錄",
        "credit_line_utilization": "信用額度使用率",
        "debt_to_income_log": "負債收入比（對數）",
        "age_bucket": "年齡區間",
    }
    return mapping.get(name, name)


def _format_basic_report(borrower_data: dict, prediction: dict, shap_top_features: list) -> str:
    score = prediction["risk_score"]
    risk_level = prediction["risk_level"]
    rows = []
    for name, shap_value, raw_value in shap_top_features[:3]:
        rows.append(
            f"- **{_feature_name_zh(name)}**：原始值 `{raw_value}`，SHAP 貢獻 `{float(shap_value):.4f}`"
        )
    feature_text = "\n".join(rows) if rows else "- 暫無可用的 SHAP 特徵資料"
    return (
        "## 借款人摘要\n"
        f"- 本次評估使用表格、時序、圖網路與文字四種模態進行融合判斷。\n"
        f"- 年齡：`{borrower_data.get('age', 'N/A')}`，月收入：`{borrower_data.get('MonthlyIncome', 'N/A')}`。\n\n"
        "## 風險評分與等級\n"
        f"- 風險評分：**{score:.1%}**\n"
        f"- 風險等級：**{risk_level}**\n\n"
        "## 主要風險因子（前 3 名 SHAP 特徵的白話解釋）\n"
        f"{feature_text}\n\n"
        "## 建議行動\n"
        "- 建議信貸人員複核逾期紀錄、信用使用率與收入負債狀況。\n"
        "- 若屬中高風險，建議補充收入證明、還款來源與貸款用途說明。"
    )


def _render_single_result(result: dict, borrower_data: dict, loan_description: str) -> None:
    score = result["risk_score"]
    risk_level = result["risk_level"]

    st.subheader("風險評分")
    col_score, col_level = st.columns(2)
    with col_score:
        st.metric("風險評分", f"{score:.1%}")
    with col_level:
        st.metric("風險等級", risk_level)
        st.markdown(f"<span style='color:{_risk_color(risk_level)}'>{risk_level}</span>", unsafe_allow_html=True)
    st.progress(score)

    st.subheader("各模態貢獻")
    st.caption("模態信號強度")
    st.altair_chart(_plot_embedding_norms(_embedding_norms(result["embeddings"])), width="stretch")

    st.subheader("SHAP 特徵解釋")
    shap_path = FIGURES_DIR / "shap_summary_day1.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="基於 XGBoost 的特徵重要性分析")
    else:
        st.warning("SHAP 圖檔尚未產生")

    st.markdown("**本筆主要特徵（中文說明）**")
    shap_rows = [
        {
            "特徵": _feature_name_zh(name),
            "原始值": raw_value,
            "SHAP貢獻": float(shap_value),
        }
        for name, shap_value, raw_value in result["top_shap_features"][:3]
    ]
    if shap_rows:
        st.dataframe(pd.DataFrame(shap_rows), width="stretch")

    st.subheader("AI 信用分析報告")
    st.caption("Powered by Claude Opus 4.7 — 報告內容基於 SHAP 計算結果生成，不包含模型推測。")
    prediction = {"risk_score": score, "risk_level": risk_level}
    try:
        report = generate_credit_report(borrower_data, prediction, result["top_shap_features"])
        if report.startswith("[報告生成失敗]"):
            report = _format_basic_report(borrower_data, prediction, result["top_shap_features"])
            st.info("未連線至 AI 服務，已顯示基本評估。")
    except Exception:
        report = _format_basic_report(borrower_data, prediction, result["top_shap_features"])
        st.info("未連線至 AI 服務，已顯示基本評估。")
    st.markdown(report)
    st.download_button(
        "下載信用分析報告",
        data=report,
        file_name="credit_risk_report.md",
        mime="text/markdown",
    )


def _style_risk_column(value: str) -> str:
    return f"color: {_risk_color(value)}"


with st.sidebar:
    st.header("資料上傳")
    uploaded = st.file_uploader("上傳借款人資料 (CSV)", type=["csv"])
    if "use_sample_data" not in st.session_state:
        st.session_state.use_sample_data = False
    if uploaded is not None:
        st.session_state.use_sample_data = False
    sample_btn = st.button("載入範例資料")
    if sample_btn:
        st.session_state.use_sample_data = True
        st.success("已載入範例資料")
    loan_description = st.text_area(
        "借款用途說明",
        value="申請人尋求資金用於日常財務管理，收入穩定，具備還款能力。",
        height=100,
    )
    run_btn = st.button("執行風險評估", type="primary")

try:
    with st.spinner("載入模型中..."):
        models = load_models()
except FileNotFoundError:
    st.error("模型檔案未找到，請先執行訓練流程")
    st.stop()

if run_btn or sample_btn:
    start_time = time.perf_counter()
    use_sample = st.session_state.use_sample_data
    batch_mode = uploaded is not None or use_sample
    if use_sample:
        sample_path = DEMO_INPUT_PATH if DEMO_INPUT_PATH.exists() else LEGACY_DEMO_INPUT_PATH
        if not sample_path.exists():
            st.error("範例資料不存在，請先建立 data/demo_input.csv")
            st.stop()
        input_df = _prepare_tabular(pd.read_csv(sample_path))
    elif uploaded is not None:
        input_df = _prepare_tabular(pd.read_csv(uploaded))
    else:
        input_df = pd.read_parquet(PROCESSED_DIR / "val.parquet").head(1)

    results = []
    detailed_results = []
    for idx, row in input_df.iterrows():
        result = predict_single_borrower(row, loan_description, models)
        borrower_id = row.get("borrower_id", idx)
        results.append(
            {
                "borrower_id": borrower_id,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
            }
        )
        detailed_results.append((row, result))

    result_df = pd.DataFrame(results)

    if batch_mode:
        st.subheader("批次評估結果")
        st.dataframe(
            result_df.style.map(_style_risk_column, subset=["risk_level"]),
            width="stretch",
        )
        st.download_button(
            "下載完整評估結果",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="credit_risk_results.csv",
            mime="text/csv",
        )

    first_row, first_result = detailed_results[0]
    _render_single_result(first_result, first_row.to_dict(), loan_description)
    st.caption(f"完成時間：{time.perf_counter() - start_time:.2f} 秒")
else:
    st.info("請上傳 CSV 或使用預設驗證樣本，然後按下「執行風險評估」。")
