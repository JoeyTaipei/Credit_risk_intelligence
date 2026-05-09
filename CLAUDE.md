# Multi-Modal Credit Risk Intelligence System

## Project Identity
- **Name:** Multi-Modal Credit Risk Intelligence System
- **Purpose:** Portfolio project for Cathay Life Insurance AI DS role
- **Demo deadline:** 2026-05-13
- **Owner:** Joey Wu (巫佳樺), Georgia Tech OMSA candidate

## Current State
- Day 1 ✅: XGBoost baseline AUC 0.85+, SHAP done
- Day 2 ✅: LSTM encoder AUC 0.72
- Day 3 ✅: GraphSAGE AUC 0.74, 840 nodes
- Day 4 ✅: sentence-BERT embeddings
- Day 5 ✅: Fusion model + Streamlit working
- Day 6 🔄: PPT in progress
- Day 7 ⬜: Rehearsal

## Architecture
- `src/data/preprocess.py`: clean tabular + synthetic time series
- `src/models/xgb_baseline.py`: XGBoost + leaf embeddings
- `src/models/lstm_encoder.py`: LSTM (input=4, output=32)
- `src/models/gnn_encoder.py`: GraphSAGE (input=5, output=32)
- `src/models/text_encoder.py`: frozen sentence-BERT → 32
- `src/models/fusion.py`: late fusion MLP (128→1)
- `src/inference/predict.py`: single borrower prediction
- `app/streamlit_app.py`: dashboard UI
- `src/utils/openai_report.py`: Claude Opus report generator

## Key Numbers (cite these exactly)
- Dataset: 150K borrowers, 6.7% default rate
- XGBoost AUC: 0.85+
- LSTM AUC: 0.72 (synthetic)
- GNN AUC: 0.74 (synthetic)
- Graph: 840 nodes, 10408 edges, avg degree 12.39
- Demo predictions: 50.7% / 48.2% / 61.4%
- Tests: 19/19 passing

## Session Rules
1. Read this file before every response
2. One task per session — finish it completely before next
3. Always comment WHY not just WHAT in code
4. After every code block add 面試講法 (zh-TW, 3 sentences max)
5. If asked about a concept, explain with a business analogy first
6. Never suggest adding new features — demo deadline is priority
7. If something will take >2 hours, suggest a simpler version first

## Business Framing
This project answers ONE question for the interviewer:
**"Can this person build AI systems that solve real financial problems?"**

Every technical decision must connect back to business value:
- XGBoost + SHAP → 法規合規, 信貸人員可解釋
- LSTM → 早期預警, 抓還款行為惡化
- GNN → 集團詐騙偵測
- GenAI report → 減少信貸人員人工撰寫時間

## What NOT to Do
- Don't add new models or modules after Day 5
- Don't refactor working code unless it blocks demo
- Don't spend time on MLOps infrastructure (Docker etc) until Day 7 buffer
