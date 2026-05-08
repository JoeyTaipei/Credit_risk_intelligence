from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

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
    st.bar_chart(_embedding_norms(result["embeddings"]))

    st.subheader("SHAP 特徵解釋")
    shap_path = FIGURES_DIR / "shap_summary_day1.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="基於 XGBoost 的特徵重要性分析")
    else:
        st.warning("SHAP 圖檔尚未產生")

    st.subheader("AI 信用分析報告")
    prediction = {"risk_score": score, "risk_level": risk_level}
    try:
        report = generate_credit_report(borrower_data, prediction, result["top_shap_features"])
        if report.startswith("[報告生成失敗]"):
            st.warning("AI報告生成失敗，顯示基本評估")
    except Exception:
        report = f"[報告生成失敗] 風險評分: {score:.1%} | 風險等級: {risk_level}"
        st.warning("AI報告生成失敗，顯示基本評估")
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
    loan_description = st.text_area("貸款申請說明", value=DEFAULT_DESCRIPTION, height=160)
    run_btn = st.button("執行風險評估", type="primary")

try:
    with st.spinner("載入模型中..."):
        models = load_models()
except FileNotFoundError:
    st.error("模型檔案未找到，請先執行訓練流程")
    st.stop()

if run_btn:
    start_time = time.perf_counter()
    if uploaded is not None:
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

    if uploaded is not None:
        st.subheader("批次評估結果")
        st.dataframe(result_df.style.applymap(_style_risk_column, subset=["risk_level"]), use_container_width=True)
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
