# 信用風險智能評估系統
### Credit Risk Intelligence System

> 一套以表格集成模型為核心、搭配多模態原型架構的信用違約預測系統。核心表格模型在 150K 真實借款人資料上達到 OOF AUC **0.866**；多模態融合模組（時序、圖、文字）為架構驗證階段的實驗性原型。

---

## 專案概覽

本系統分為兩個層次：

**① 生產級核心模型（Validated Core）**
在完整 GiveMeSomeCredit 資料集（150K 筆）上，以 Optuna 調優 XGBoost、LightGBM、CatBoost，搭配 Stratified 5-Fold CV 與加權 Ensemble，達到可信賴的 OOF AUC 0.866。

**② 實驗性多模態原型（Experimental Prototype）**
展示如何將時序（Bi-LSTM）、關係圖（GraphSAGE）、申請文字（sentence-BERT）整合進 Late Fusion 架構。各模態目前使用不同資料來源，尚未完成借款人級別對齊，Fusion AUC 僅代表架構可行性，不代表生產預測能力。

---

## 模型效能總覽

| 模組 | 方法 | 資料 | AUC | 備注 |
|---|---|---|---|---|
| **表格（核心）** | XGBoost | GiveMeSomeCredit 150K | 0.8649 | Optuna 調優，5-Fold OOF |
| **表格（核心）** | LightGBM | GiveMeSomeCredit 150K | 0.8652 | 同上 |
| **表格（核心）** | CatBoost | GiveMeSomeCredit 150K | 0.8657 | 同上 |
| **表格（核心）** | **三模型 Ensemble** | GiveMeSomeCredit 150K | **0.8658** | CatBoost 80% 權重 |
| 時序（原型）| Bi-LSTM 合成 | Bernoulli 合成序列 | 0.72 | 已升級，列為參考 |
| **時序（原型）** | **Bi-LSTM 真實** | Lending Club 100K | **0.97** | 序列相關性驗證；與表格資料集不同 |
| 圖（原型）| GraphSAGE | 合成 Cosine 相似圖 | 0.74 | 非真實擔保關係 |
| 文字（原型）| frozen sentence-BERT | 合成貸款描述 | — | 無獨立監督標籤 |
| **Fusion（原型）** | Late Fusion MLP | 混合，未對齊 | **0.74** | 架構驗證，非生產指標 |

---

## 核心表格模型（Validated Core）

### 資料
- **來源：** Kaggle GiveMeSomeCredit（`cs-training2.csv`）
- **規模：** 149,999 筆，正例率 6.7%
- **目標變數：** `SeriousDlqin2yrs`（未來兩年 90 天以上逾期）

### 方法
```
資料清洗（哨兵碼 96/98、MonthlyIncome 缺失值處理）
→ 特徵工程（total_past_due、delinq_acceleration 等 14 個衍生特徵）
→ Optuna 超參數搜尋（100 trials，最大化 3-Fold AUC）
→ Stratified 5-Fold CV（StratifiedKFold, random_state=42）
→ XGBoost + LightGBM + CatBoost 三模型
→ scipy.optimize 加權 Ensemble
```

### 結果
```
XGBoost  OOF AUC: 0.8649
LightGBM OOF AUC: 0.8652
CatBoost OOF AUC: 0.8657
Ensemble OOF AUC: 0.8658  ← 核心指標
Fold 穩定性: 最低 0.858，最高 0.870，波動 < 0.012
```

### 重要檔案
```
src/data/features_v2.py              特徵工程
src/training/train_xgb_v2.py        XGBoost Optuna + 5-Fold
src/training/train_lgbm_catboost.py LightGBM + CatBoost
src/training/train_ensemble.py      加權 Ensemble
data/processed/xgb_best_params.json 最佳超參數
data/processed/ensemble_weights.json Ensemble 權重
data/processed/xgb_oof.npy          OOF 預測（149,999 筆）
```

---

## 實驗性多模態原型（Experimental Prototype）

### 架構說明

四個模態各自編碼為 32 維向量，在 Late Fusion MLP 層拼接為 128 維，輸出違約機率。

```
Tabular  → XGBoost Leaf Embedding  → 32-dim
Time Series → Bi-LSTM              → 32-dim  ┐
Graph       → GraphSAGE            → 32-dim  ├→ 128-dim → MLP → P(default)
Text        → sentence-BERT        → 32-dim  ┘
```

### 各模態現況

| 模態 | Encoder | 資料來源 | 現況 |
|---|---|---|---|
| 表格 | XGBoost Leaf Embedding | GiveMeSomeCredit 1,200筆 | ⚠️ 使用舊版小樣本 |
| 時序 | Bi-LSTM | Lending Club 100K | ✅ 真實資料，序列相關性驗證 |
| 圖 | GraphSAGE（2層 SAGEConv）| Cosine 相似度合成圖 | ⚠️ 非真實擔保關係 |
| 文字 | frozen MiniLM + 投影頭 | 模板生成 5,000 筆 | ⚠️ 語意多樣性有限 |

### 時序模態升級紀錄

```
原版（Bernoulli 合成）:
  - 每月逾期事件獨立抽樣，無序列相關性
  - Val AUC: 0.72（反映合成邏輯，非真實行為）

升級版（Lending Club 真實資料）:
  - 100,000 筆真實貸款，正例率 3.4%
  - 用前 10 個月預測第 11–12 個月（嚴格時間切割）
  - 序列相關性驗證：90% 違約者在違約前 1–5 個月出現逾期前兆
  - Val AUC: 0.97
  - 注意：與 GiveMeSomeCredit 為不同資料集，借款人 ID 未對齊
```

---

## 已知限制與模態對齊問題

### 核心限制

**借款人 ID 未對齊：**
現階段四個模態使用不同資料集，Fusion 層拼接的是「來自不同借款人群體的向量」，而非同一批借款人的多角度表示。

```
XGBoost Embedding  → GiveMeSomeCredit 借款人（1,200筆舊版）
LSTM Embedding     → Lending Club 借款人（不同資料集）
GraphSAGE          → 合成相似圖節點
Text Embedding     → 合成貸款描述
```

因此，**Fusion AUC 0.74 是架構可行性的驗證指標，不是系統整體預測能力的代表數字。**

### 其他限制

- 圖模型的邊代表特徵 Cosine 相似度，不代表真實擔保或共同借款關係
- 文字模板生成，sentence-BERT Embedding 因句型相似而聚集，區辨能力被低估
- 表格 Encoder 的 Leaf Embedding 仍使用 1,200 筆舊版資料生成

---

## 未來工作：對齊的多模態 Fusion

完成借款人級別對齊需要以下步驟：

```
Step 1: 用 cs-training2.csv (150K) 重新生成 XGBoost Leaf Embeddings
        → 使用 5 個 fold 模型的 model.apply() 輸出

Step 2: 選擇時序對齊策略
        Option A: 用 cs-training2.csv 靜態特徵重新合成時序
                  → 用 lstm_encoder_lc.pt 編碼（Markov Chain 改善序列相關性）
        Option B: 取得與 GiveMeSomeCredit 借款人重疊的真實時序資料

Step 3: 重建合成圖（用 150K 借款人）

Step 4: 重訓 Fusion MLP（訓練樣本從 840 升至 ~120K）
        → 預期 Fusion AUC 明顯提升
```

---

## 可解釋性設計

```
特徵層 SHAP  → 哪個欄位推高了風險、推高多少（XGBoost 層）
模態層 SHAP  → 四種資料來源，哪個主導了這次判定（Fusion 層）
GenAI 報告   → Claude API 將 SHAP 輸出翻譯為中文授信摘要
             → Prompt 限制：只能使用 SHAP 已算好的事實
```

---

## 合規設計

| 功能 | 說明 |
|---|---|
| Adverse Action | SHAP 直接輸出具體拒貸原因，符合金融消費者保護法要求 |
| Fairness Audit | 定期分群確認核准率無系統性偏差 |
| Audit Trail | 記錄 `model_version`、`shap_top_features`、`override_reason`、`reviewer_id` |

---

## 技術棧

| 分類 | 技術 |
|---|---|
| 表格模型 | XGBoost、LightGBM、CatBoost |
| 超參數調優 | Optuna |
| 深度學習 | PyTorch（LSTM、Fusion MLP）|
| 圖神經網路 | PyTorch Geometric（GraphSAGE）|
| 自然語言處理 | sentence-transformers（frozen MiniLM）|
| 可解釋性 | SHAP |
| 生成式 AI | Anthropic Claude API |
| 視覺化 | Streamlit、PowerBI |
| 實驗追蹤 | MLflow |

---

## 如何執行

```bash
# 安裝相依套件
pip install -e .

# 核心表格模型（完整流程）
python -m src.training.train_xgb_v2 --data-path data/raw/cs-training2.csv --n-trials 100
python -m src.training.train_lgbm_catboost
python -m src.training.train_ensemble

# 時序模型（Lending Club）
python -m src.data.lending_club_timeseries
python -m src.training.train_lstm --data_source lending_club

# 多模態 Fusion（原型）
python -m src.training.train_fusion

# Dashboard
streamlit run app/streamlit_app.py
```

---

## 作者

**Joey Wu（巫佳樺）**
Georgia Tech OMSA
