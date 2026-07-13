# SiliconPulse AI

**Predicting next-month semiconductor spot prices by fusing market momentum, supply-chain fundamentals, and geopolitical trade risk into a single machine learning pipeline.**

🔗 **Live app:** [semiconductor-price-predictor.streamlit.app](https://semiconductor-price-predictor-7nx9azlczvuwrgpsd7d8cq.streamlit.app/)
📦 **Repo:** [github.com/Moskowdigama/semiconductor-price-predictor](https://github.com/Moskowdigama/semiconductor-price-predictor)

---

## The problem

Standard semiconductor price forecasts lean almost entirely on autoregressive historical price curves — yesterday's price predicting tomorrow's. That approach is blind to the things that actually move this market: a new export control, a wafer fab coming online, an AI chip demand spike. SiliconPulse AI builds a single feature matrix that captures all three at once — price momentum, supply-and-demand fundamentals, and unstructured trade policy text — and feeds them into one model.

---

## Data architecture

The core time-series dataset was expanded with three industry datasets, each aggregated **annually** before merging into the core dataframe — this was a deliberate choice to prevent row duplication during the relational merge.

| Dataset | Purpose | Engineered features |
|---|---|---|
| `fab_capacity.csv` | Supply-side infrastructure readiness | Global `monthly_wafer_capacity`, aggregated per year into a factory-volume constraint index |
| `ai_chip_market.csv` | Demand-side pressure from the AI boom | Annual sums of `estimated_shipments_units` and `estimated_revenue_usd_m` |
| `chip_companies_financials.csv` | Forward-looking investment signal + trailing financial health | Aggregated `capex_usd_bn` (leading indicator) and average `operating_margin_pct` (trailing indicator) |

---

## Feature engineering

The final pipeline maps everything into a **113-dimensional feature matrix**:

- **NLP on trade policy text** — Raw regulatory and policy strings are run through a fitted `TfidfVectorizer` (English stop words removed, top 100 tokens by weight kept), converting qualitative geopolitical risk into quantitative features.
- **Time-series momentum** — `price_lag_1`, `price_lag_2`, and a `price_rolling_mean_3m` window preserve the sequential direction of the spot market instead of treating each row independently.
- **Category vectorization** — Hardware types (from consumer NAND to server memory modules) are one-hot encoded into binary dimensions.

---

## Validation strategy

Random shuffling would leak future prices into training here, since the data is grouped by product type across time blocks. To prevent that:

1. The full matrix is sorted globally by `year_month`.
2. A strict chronological split is applied — the model trains on the first 80% of history and is evaluated only on the final 20%, across all chip models simultaneously.

## Model

- **Algorithm:** Random Forest Regressor, 100 trees, parallelized
- **Out-of-sample R²:** 0.9766

---

## Deployment

The trained pipeline was decoupled from the Kaggle notebook and serialized with `joblib` into four production assets:

| File | Contents |
|---|---|
| `macro_model.pkl` | The trained Random Forest ensemble |
| `tfidf.pkl` | The fitted TF-IDF vocabulary/weights |
| `product_features.pkl` | The exact column order required to reconstruct one-hot categorical inputs |
| `macro_defaults.pkl` | Static baseline macro values, so the UI has grounded defaults |

Deployed via GitHub → Streamlit Community Cloud.

### Issues hit during deployment (and how they were fixed)

- **Path mismatch** — the app script pointed to an `assets/` subfolder, but the `.pkl` files were sitting at repo root. Fixed by loading directly from root.
- **Dependency spawn errors** — pinning exact versions (`==`) in `requirements.txt` conflicted with Streamlit Cloud's runtime and corrupted the environment during resolution. Fixed by unpinning versions and letting the package manager resolve them:
- streamlit
scikit-learn
pandas
numpy
joblib

---

## Tech stack

Python · pandas · scikit-learn (Random Forest, TF-IDF) · joblib · Streamlit · GitHub → Streamlit Community Cloud

---

## What's next

- Expand the trade-policy text corpus beyond the current scenario set
- Add prediction intervals rather than a single point estimate
- Explore gradient boosting as a comparison baseline against the Random Forest
