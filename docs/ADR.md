# Architecture Decision Record

**System:** Multi-Modal Credit Risk Intelligence
**Author:** Joey Wu
**Date:** 2026-05-08
**Status:** Accepted (Demo v1)

---

## Decision 1 — Late Fusion over Early Fusion

**What:** Modality-specific encoders (XGBoost, LSTM, GraphSAGE, sentence-BERT) each produce their own fixed-size embedding vector, which are concatenated at the classifier layer after independent encoding — not merged into a single flat feature vector before any learning occurs.

**Why:** Each modality has an incommensurable feature space and optimal preprocessing pipeline; forcing raw tabular columns, time steps, graph node features, and token sequences into a single input vector destroys structural information and makes the pipeline brittle to missing modalities, a practical concern when a new borrower has no loan history or no social graph.

**Alternative considered:** Early fusion (concatenate all raw features before any encoder) was rejected because it requires a single model to simultaneously learn tabular interactions, sequential dependencies, graph topology, and semantic meaning — a representational burden that degrades performance on each individual task.

---

## Decision 2 — GraphSAGE over GAT

**What:** The graph encoder uses GraphSAGE (inductive, neighborhood-sampling aggregation) rather than a Graph Attention Network to embed co-borrower relationship graphs into fixed-size node representations.

**Why:** GraphSAGE is inductive by design — it generalizes to unseen nodes at inference time without retraining, which is essential in production where new borrowers join daily; its neighborhood-sampling approach also keeps memory usage tractable on large graphs, whereas full-graph attention becomes prohibitive at scale.

**Alternative considered:** GAT was considered for its learned, attention-weighted neighbor aggregation, which could capture heterogeneous relationship importance, but GAT's transductive default and O(N²) attention complexity make it a worse fit for a production-trajectory design, and the added interpretability benefit is already covered by SHAP at the fusion layer.

---

## Decision 3 — Frozen sentence-BERT over Fine-Tuned BERT

**What:** The text encoder uses a pre-trained sentence-BERT model with all weights frozen, extracting 768-dimensional sentence embeddings from synthetic loan descriptions without any gradient updates during training.

**Why:** Fine-tuning BERT requires labeled text paired with default outcomes and non-trivial GPU time — neither available in a 7-day sprint with synthetic text; frozen sentence-BERT still captures rich semantic structure, and the downstream MLP fusion classifier can learn to extract credit-relevant signal from the embedding space.

**Alternative considered:** Domain-adapted FinBERT was considered for its financial-text pre-training, but FinBERT targets sentiment and classification over financial news rather than loan narrative embeddings, and sentence-BERT's sentence-level pooling is more directly useful for our embedding-and-fuse architecture; FinBERT fine-tuning on labeled loan text is the correct production path.

---

## Decision 4 — LSTM over Transformer for Time Series

**What:** A bidirectional LSTM encodes each borrower's monthly payment history (12–36 time steps) into a fixed embedding rather than a Transformer-based architecture such as Temporal Fusion Transformer.

**Why:** At sequence lengths of 12–36 steps, LSTM matches Transformer performance without the positional encoding tuning, multi-head attention overhead, and variable-selection network complexity that TFT requires; in a 7-day sprint, LSTM's simpler hyperparameter space (hidden size, layers, dropout) is faster to integrate and debug in a late-fusion pipeline.

**Alternative considered:** Temporal Fusion Transformer was evaluated for its interpretable temporal attention weights, but TFT's engineering overhead — gating layers, variable selection, static covariate encoders — is disproportionate to synthetic sequences of under 40 steps and would consume roughly two days of the sprint for marginal benefit on this sequence regime.

---

## Decision 5 — Synthetic Data Acceptable for Portfolio

**What:** Three of four modalities — monthly payment time series, co-borrower graphs, and loan description text — are synthetically generated rather than sourced from real borrower records.

**Why:** Real consumer financial data is proprietary, subject to Taiwan's Personal Data Protection Act and equivalent obligations, and unavailable in an academic-portfolio setting; synthetic data allows full end-to-end pipeline demonstration without legal exposure, and the architecture — not the performance numbers — is the interview deliverable.

**Alternative considered:** Restricting the demo to the single real modality (GiveMeSomeCredit tabular data) was rejected because a single-modality system cannot demonstrate multi-modal fusion, which is the core technical claim of the project; all performance metrics reported on synthetic modalities are explicitly labeled as artifacts of the data generation process and carry no production validity.

---

## Production Migration Notes

If Cathay provided real data tomorrow, four things would change immediately. First, the text encoder would be replaced or fine-tuned: real loan application narratives would warrant FinBERT or a domain-adapted model trained on a labeled default corpus, not frozen sentence-BERT on synthetic descriptions. Second, the graph would be rebuilt from actual co-borrower, guarantor, and referral relationships stored in Cathay's CRM — requiring graph construction and temporal edge handling that the synthetic random graph does not model. Third, the LSTM would be retrained with strict temporal train/validation splits on real repayment histories to prevent look-ahead bias, a subtle issue that synthetic generation sidesteps but real data enforces. Fourth, XGBoost would be recalibrated on Cathay's internal feature distribution; GiveMeSomeCredit represents a U.S. consumer population with different delinquency base rates and feature definitions than a Taiwanese insurance portfolio. Additionally, all SHAP explanations and GPT-4o-mini narrative reports surfaced to loan officers would require legal and compliance review before production deployment, as automated credit explanations are regulated under Taiwan's financial consumer protection framework.
