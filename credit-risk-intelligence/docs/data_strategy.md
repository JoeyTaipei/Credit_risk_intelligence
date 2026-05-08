# Data Strategy

**System:** Multi-Modal Credit Risk Intelligence
**Date:** 2026-05-08

---

### 1. Tabular

- **Source:** Real — GiveMeSomeCredit (Kaggle, 150,000 U.S. borrowers)
- **Limitations:**
  - U.S. consumer population; delinquency base rates and feature distributions do not transfer directly to a Taiwanese insurance portfolio.
  - Features are pre-aggregated snapshots (e.g., `NumberOfTimes90DaysLate`); no raw transaction-level data is available, which limits temporal modeling from this source alone.
- **Interview defense:** It is a widely benchmarked, well-documented real credit dataset that grounds the tabular encoder in genuine default signal, giving the project at least one modality with production-credible labels.

---

### 2. Time Series

- **Source:** Synthetic — monthly payment sequences derived from GiveMeSomeCredit tabular features
- **Construction logic:**
  - Assign each borrower a sequence length drawn from U(12, 36) months.
  - Map the borrower's `NumberOfTimes30-59DaysLate`, `60-89DaysLate`, and `90+DaysLate` counts to a per-month delinquency probability.
  - Sample each month's payment status (`on-time`, `late-30`, `late-60`, `default`) from a multinomial distribution parameterized by those probabilities.
  - Inject small Gaussian noise on a continuous "payment ratio" feature (amount paid / amount due) to simulate partial payments.
  - Defaulted borrowers (target = 1) receive a monotonically increasing late-payment trend in the final 6 months to embed a detectable signal.
- **Limitations:**
  - The delinquency trend is hand-engineered, so the LSTM will learn a synthetic artifact rather than a real behavioral pattern; reported sequence-model metrics are not meaningful outside this demo.
  - No seasonality, macroeconomic shocks, or borrower-level behavioral drift are modeled — real payment histories are far noisier and more complex.
- **Interview defense:** The construction is deliberately designed so that default signal exists in the sequence, which lets the LSTM demonstrate that it can learn temporal patterns — the goal is architectural integration, not production accuracy.

---

### 3. Graph

- **Source:** Synthetic — borrower similarity graph derived from normalized tabular features
- **Construction logic:**
  - L2-normalize all tabular features per borrower.
  - Compute pairwise cosine similarity across all borrower pairs.
  - Draw a directed edge between borrower i and borrower j if cosine similarity ≥ 0.85.
  - Cap each node's out-degree at 10 neighbors (k-NN cap) to prevent hub collapse.
  - Assign edge weights equal to the raw cosine similarity score.
- **Limitations:**
  - Cosine similarity on normalized features is a proxy for financial similarity, not a real social or co-borrower relationship; the graph has no semantic meaning that would exist in a production CRM.
  - The 0.85 threshold is arbitrary — it was tuned to produce a connected graph of manageable density, not derived from any domain knowledge about borrower relationships.
- **Interview defense:** The graph construction is fully reproducible and rule-based, which is honest — it exists to demonstrate that GraphSAGE can embed relational structure into the fusion pipeline, not to claim the edges represent real-world connections.

---

### 4. Text

- **Source:** Synthetic — loan purpose descriptions generated via Faker and string templates
- **Construction logic:**
  - Assign each borrower a loan purpose category (home improvement, debt consolidation, medical, auto, education, small business) sampled proportionally to GiveMeSomeCredit feature distributions.
  - Fill a purpose-specific sentence template with borrower attributes (loan amount bucket, employment length, age range) using Faker-generated names and plausible figures.
  - Append a one-sentence risk qualifier for high-risk borrowers (target = 1) to embed weak default signal in the text.
- **Limitations:**
  - Template-generated text has no linguistic variation beyond surface-level substitution; sentence-BERT embeddings will cluster tightly by template, not by genuine semantic content.
  - No real underwriter notes, income verification language, or applicant-written narratives are present — the text modality is the furthest from what a production system would actually process.
- **Interview defense:** The templates produce grammatically valid, domain-plausible sentences that give sentence-BERT something real to embed, which is sufficient to demonstrate that a text encoder can be wired into a late-fusion pipeline.

---

## Why Three of Four Modalities Are Synthetic — and That Is OK

Real banks already have all four modalities: transaction ledgers give time series, CRM systems give borrower relationship graphs, loan applications give text, and internal scoring models give structured tabular features. The synthetic construction here is a portfolio limitation, not a design flaw — I do not have access to proprietary financial data, and fabricating it honestly is the only legal and ethical alternative to real records. The goal of this project is to demonstrate that I can architect a multi-modal fusion pipeline, select appropriate encoders for each data type, and integrate explainability tooling end-to-end. Any performance number reported on the synthetic modalities reflects the quality of the data generator, not the model — I will say exactly that in an interview, because an interviewer who works with real data every day will respect that honesty far more than inflated claims.
