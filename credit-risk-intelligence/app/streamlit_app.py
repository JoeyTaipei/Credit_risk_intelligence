import streamlit as st

st.set_page_config(page_title="信用風險智能評估", layout="wide")
st.title("信用風險智能評估系統")
st.caption("信用風險智能 — 多模態 AI 流程")

# Sidebar
with st.sidebar:
    st.header("資料上傳")
    uploaded = st.file_uploader("上傳借款人資料 (CSV)", type=["csv"])
    run_btn = st.button("執行風險評估", type="primary")

# Main layout: 4 placeholder sections
col1, col2 = st.columns(2)
with col1:
    st.subheader("風險評分")
    score_placeholder = st.empty()
    st.subheader("特徵重要性")
    importance_placeholder = st.empty()
with col2:
    st.subheader("SHAP 解釋")
    shap_placeholder = st.empty()
    st.subheader("AI 信用分析報告")
    report_placeholder = st.empty()

if run_btn:
    st.info("模型尚未載入 — Day 5 交付項目")
