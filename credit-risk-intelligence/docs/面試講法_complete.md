# 面試講法 — Multi-Modal Credit Risk Intelligence System

**場合：** 國泰人壽 AI Data Scientist 面試，8 分鐘簡報 + Q&A
**原則：** 口語自然，像跟同事討論；不要學術腔；誠實面對合成資料的限制

---

## Day 1：資料清洗與特徵工程

- 資料來自 GiveMeSomeCredit（Kaggle，15 萬筆美國借款人紀錄），目標變數違約率約 6.7%，是典型的不平衡資料集，因此選用 `scale_pos_weight ≈ 14` 來調整 XGBoost 的損失函數權重，而非刪除樣本或過採樣。
- 清洗最棘手的部分是逾期欄位裡的哨兵碼 96 和 98——這是信用局的「不適用」代碼，不是真實的逾期次數，如果不處理模型會把這些行當成「逾期 96 次的超高風險借款人」，AUC 數字看起來正常但推論邏輯完全錯誤。
- **「96 和 98 是新手最容易漏掉的陷阱，因為 `df.describe()` 只顯示 max=98，看起來像個大數字，但不查資料字典、不畫分佈圖的話，永遠不會意識到它是特殊編碼——這正是 EDA 要做細的原因。」**
- `MonthlyIncome` 缺失率約 20%（EDA 實際測量值，不是 5%），用分組中位數填補而非均值，原因是分佈極度右偏，少數超高收入者會把均值拉到不代表典型借款人的水平。
- 特徵工程新增 `total_past_due`（三個逾期欄位加總）和 `has_any_delinquency`（二元旗標），這兩個特徵最終在 XGBoost SHAP 分析中進入 top 5，證明聚合設計是正確的。

---

## Day 2：LSTM 時序模組

- 靜態 tabular 快照只能告訴你「借款人目前有幾次逾期」，但看不到趨勢——同樣是 2 次逾期，一個是 3 年前的舊帳、另一個是最近三個月連續發生，風險截然不同；時序資料把「什麼時候發生了什麼」這個維度還原回來。
- 合成時序是從 tabular 特徵衍生的：用 `NumberOfTime30-59DaysPastDueNotWorse` 推算每月違約機率、用 `DebtRatio` 和 `MonthlyIncome` 建構餘額壓力代理，每個借款人模擬 12 個月的 [utilization, payment_ratio, is_late, balance] 序列。
- **「LSTM 的 last hidden state 是一個 learned sequence summary——gate 機制在 forward pass 已經決定哪些時間步的資訊值得記住、哪些可以遺忘，直接拿這個 summary 做 late fusion 比 mean pooling 更能保留序列的因果結構，這正是選 last hidden state 而非 output mean 的理由。」**
- 合成時序的限制是 `is_late_t` 每個月獨立抽樣，沒有序列相關性，但真實逾期事件往往是連環發生的；Validation AUC 0.72 反映的是 data generator 的統計邏輯，不是真實還款行為。
- 真實場景升級路徑：把 `create_synthetic_time_series` 換成真實月度還款紀錄的 loader，`LSTMEncoder` 的架構和 `embedding_dim=32` 介面完全不需要改動——模組化設計的優點就在這裡。

---

## Day 3：GraphSAGE GNN 模組

- 圖資料捕捉的是「借款人在財務群體中的位置」——高風險借款人傾向聚集在同一個特徵空間裡；GNN 最大的商業價值是找出傳統信用評分看不到的群聚詐欺，一個擔保人圈子裡的借款人風險往往一起爆。
- 圖是合成的：把 5 個 tabular 特徵正規化後算 cosine similarity，相似度 ≥ 0.85 就連邊，每個節點最多 10 個鄰居；cosine 比 euclidean 更適合這個用途，因為它測量「方向相似性」，兩個月收入差距大但財務行為模式相同的借款人，cosine similarity 會很高但 euclidean distance 會很大。
- **「SAGEConv 最關鍵的優勢是 inductive learning——它學習的是一個可以應用到任意鄰居集合的聚合函數，新借款人申請入件時不需要重新訓練整個圖模型，直接用已訓練的 SAGEConv 產生 embedding，這正是 GraphSAGE 在生產環境比 GCNConv（transductive）更可行的理由。」**
- 2 層 SAGEConv 的業務意義：第 1 層讓每個節點看到直接鄰居的信用特徵；第 2 層再聚合一次等於覆蓋「鄰居的鄰居」——在詐欺偵測場景，這個 2-hop 資訊能捕捉到「我的鄰居的鄰居大量違約」的風險傳染信號，Validation AUC 0.74。
- 合成圖的邊代表「特徵相似的借款人」，沒有業務語意；真實場景是從 CRM 的共同借款人、擔保人、推薦人關係建邊，那才有真正的信用傳染意義，也是這個模組真實資料下升級的第一步。

---

## Day 4：sentence-BERT 文字 Encoder

- 用 pretrained sentence-BERT 而不從頭訓練，是因為 fine-tuning 需要大量標注文本資料——我們只有 5,000 筆模板生成的合成描述，完全不足以更新 2,200 萬個 transformer 參數；sentence-BERT 已經用 10 億對句子對比訓練過，遷移這個知識比在合成資料上自訓練要靠譜得多。
- 選 `all-MiniLM-L6-v2` 不選 `bert-base`：MiniLM 是 Microsoft 用知識蒸餾把 bert-base 壓縮成 6 層的句子相似度模型，22M 參數（bert-base 是 110M），CPU 上跑 5 倍快，輸出直接是 384 維句子級 embedding，不需要額外 pooling——7 天 demo 沒有 GPU，這個選擇是工程務實考量。
- **「Frozen embedding 在資料量不足時反而比 fine-tuning 更好，因為 fine-tuning 需要足夠的監督信號才能改善 pre-trained 權重，資料量不足時只會把有用的通用語意表示覆寫成對訓練集的過擬合——frozen 讓 projection head 的 12,320 個參數在穩定的 384 維語意空間上有東西可以學，但不會破壞已有的知識。」**
- 384 維壓到 32 維的 projection 風險：8% 壓縮率可能丟失語意細節，projection head 必須在有限的監督信號下學會哪 32 個維度對違約預測有用；緩解方法是把 `embedding_dim` 放大到 64 或在 projection 後加 L2 正規化。
- 真實場景的文字來源：最直接的是**貸款申請書自述欄位**（借款目的、還款計畫），其次是**客服通話紀錄**（還款困難往往先反映在溝通內容）和**信用報告備註欄位**（催收紀錄、爭議說明）；這些非結構化文字在傳統模型裡完全被忽略，是多模態系統真正的差異化優勢。

---

## Day 5：Late Fusion 模型

- Late fusion 把四個 encoder 各自輸出的 32 維 embedding 在決策層拼接成 128 維，而不是在輸入層把原始特徵合在一起；這樣每個 encoder 能用最適合自己模態的歸納偏置獨立學習，tabular 用決策樹分裂、時序用 gate 機制、圖用鄰域聚合、文字用 attention，早期融合會破壞這些模態專屬的表示學習。
- 選 concat 不選 attention-weighted fusion（動態學習信任哪個模態），原因是 840 筆訓練樣本下，attention 的額外參數量會導致過擬合；attention 融合是真實資料下的升級路徑，不是這個 demo 的正確工程決策。
- **「這個 128 維的拼接向量在物理上代表一個借款人的完整信用輪廓：前 32 維是 XGBoost 的風險群集標記（他是什麼類型的借款人），中間 32 維是他的動態還款行為，再 32 維是他在借款人群體中的社群位置，最後 32 維是他自己描述的借款意圖——四個維度同時存在於一個向量裡，這就是 multi-modal fusion 的核心主張。」**
- Dropout=0.3 比各 encoder 的 0.2 高，因為 128 維拼接輸入更容易讓 MLP 學到某個模態的特定神經元組合，而不是真正的跨模態推理；較高的 dropout 壓力強迫模型學習對各模態 embedding 更具泛化性的組合方式。
- 這個系統對面試官證明的不是「我能 AUC 0.99」，而是「我知道怎麼把四種完全不同的資料模態整合進一個可維護的 pipeline，每個模組有清楚的介面契約，可以獨立訓練、獨立替換、獨立 debug」——這才是信貸場景下 senior DS 真正需要的能力。

---

## 自我介紹 60 秒版

我叫 Joey Wu，背景是土木工程轉 Data Science。大學念的是結構力學，轉職的契機是在工地現場做工程監造時，發現用傳統方式分析工期延誤和成本超支的效率很低，開始自學 Python 和機器學習，後來發現這個方向比鋼筋混凝土更適合我。目前我在修 Georgia Tech 的 OMSA 線上碩士，同時在做兩個自主專案：一個是您今天看到的多模態信用風險系統，把 tabular、時序、圖、文字四種資料融合成一個 late fusion 分類器，用 XGBoost、LSTM、GraphSAGE、sentence-BERT 四個 encoder 搭配 SHAP 可解釋性和 GPT-4o-mini 報告生成；另一個專案是用機器學習預測營建工程成本超支，結合我的工程背景做特徵工程。我應徵國泰人壽 AI DS 的原因是貴司在保險精算和風險評估上有大量結構化資料，也有導入 AI 的明確業務需求，這個交集正好是我最想深耕的方向。

---

## 必考題 8 問

---

### Q1. 為什麼要做 multi-modal 而不是只用 tabular？

單純 tabular 只能看到靜態快照，看不到動態行為，也看不到借款人在社交網絡裡的位置。文字資料可以捕捉申請人自述的用途，跟實際行為交叉比對能發現很多矛盾。**真實銀行本來就有這四種資料，我只是把架構做出來，證明我知道怎麼把它們整合在一起。** 當然合成資料的 AUC 數字沒有意義，但架構是真實的。如果國泰給我真實資料，這套 pipeline 不用大改就能接上去。

---

### Q2. 圖怎麼建的？真實還是合成？合成的話面試官會 challenge 嗎？怎麼回應？

圖是合成的，做法很直接：把 5 個 tabular 特徵正規化後算 cosine similarity，相似度 ≥ 0.85 就連邊，每個節點最多 10 個鄰居，邊權就是 cosine similarity 值。我自己知道這不是真實的社交或擔保人關係，它只是「特徵相似的借款人」，沒有真實的業務意義。**我想展示的是我知道 GraphSAGE 怎麼接進 late fusion pipeline，不是說這個合成圖能預測真實違約。** 真實場景是從 CRM 的共同借款人、擔保人、推薦人關係建圖，那個才有真正的信用傳染意義。面試官如果 challenge，我就直接這樣說，不要替合成圖辯護。

---

### Q3. 為什麼用 LSTM 不用 Transformer？

我的時序資料每個借款人只有 12 個月，序列非常短。**在這麼短的序列上，LSTM 跟 Transformer 的表現差不多，但 LSTM 要調的超參數少很多，7 天 sprint 裡用 LSTM 是務實的工程選擇，不是因為我認為 LSTM 技術上更好。** 如果序列是 100 步以上，或者有複雜的長距離依賴，我會換 Temporal Fusion Transformer。面試官如果問「你知道 TFT 嗎？」，答案是知道，我選 LSTM 是因為規模不值得。

---

### Q4. GNN 在信用風險問題上的真實商業價值是什麼？

傳統信用評分只看單一借款人，但詐欺或違約很少是孤立發生的。**GNN 最大的商業價值是找出傳統模型看不到的群聚詐欺：一個擔保人圈子裡的借款人風險會一起爆，這在 tabular 模型裡完全看不見。** 第二個價值是識別 high-risk cohort：高風險借款人的鄰居往往也是高風險，圖可以讓這個信號傳遞出去。第三個是跨借款人的風險傳染，供應鏈融資裡一家公司倒閉，沿著圖走就能提前標記出受影響的其他借款人。這三個場景在真實銀行裡都是 GNN 的核心應用。

---

### Q5. 如果只能保留一個 modality，你會選哪個？為什麼？

選 tabular，理由很簡單。**Tabular 是這個 demo 裡唯一有真實 label 的 modality，其他三個合成資料的 AUC 數字都是 data generator 的 artifact，不代表真實的信用預測能力。** XGBoost 在 tabular 上的表現本來就已經很強，GiveMeSomeCredit 這個資料集有 15 萬筆真實美國借款人的違約紀錄。其他 modality 的價值是在有真實資料的時候才能發揮出來。如果只有一個 modality 可以帶進國泰的真實環境，一定是 tabular。

---

### Q6. Late fusion vs early fusion 你怎麼選的？

Early fusion 是把所有特徵拼成一個大向量再丟進一個模型，問題是 tabular 的數值特徵、時序的向量、圖的 node embedding、文字的 sentence embedding 根本不在同一個特徵空間，強制合在一起只會互相干擾。**Late fusion 最重要的實務優勢是可以優雅地處理 modality 缺失：新借款人沒有歷史紀錄、或者不在圖裡，系統還是可以用剩下的 modality 給分，不會整個 pipeline 壞掉。** 每個 encoder 獨立維護也更好 debug，LSTM 壞了不影響 XGBoost。這在真實系統裡很重要，因為各個資料來源的穩定性不一樣。

---

### Q7. 模型可解釋性怎麼處理？

可解釋性分兩層。第一層是 fusion 層的 SHAP，可以看到四個 modality 各自對最終預測貢獻了多少，「是文字拉高了風險還是時序拉高了風險」這個問題在這層回答。**第二層是 XGBoost 的 feature-level SHAP，可以精確說出「這個借款人因為 90 天逾期次數太高，風險分數上升了 X 分」，這才是信貸員真正需要的解釋。** GraphSAGE 本身沒有 attention weights（那是 GAT 的東西），但可以分析鄰居節點特徵對當前節點 embedding 的影響；如果需要更細的 GNN 解釋可以換 GAT。最後 GPT-4o-mini 把 SHAP 結果翻成人讀得懂的敘事報告，prompt 裡限制它只能用 SHAP 提供的事實，不能自由發揮。

---

### Q8. 如果國泰給你真實資料，你的 pipeline 第一個會改哪裡？

**第一個換文字 modality：合成的 loan description 沒有真實的信用訊號，換成真實申請書描述之後，要 fine-tune 一個金融領域的預訓練模型，或者至少換成 FinBERT，frozen sentence-BERT 在這裡就不夠了。** 第二個換圖，不再用 cosine similarity 代理，改從 CRM 的共同借款人、擔保人、推薦人關係直接建邊，這樣 GNN 才有真實的業務意義。時序部分要重新設計 train/val/test 的 temporal split，確保沒有 look-ahead bias，這在合成資料裡不是問題，但真實資料裡這個錯誤很常見。XGBoost 也需要在國泰的特徵分佈上重新校準，GiveMeSomeCredit 是美國人的資料，基礎違約率跟台灣保險客群不一樣。
