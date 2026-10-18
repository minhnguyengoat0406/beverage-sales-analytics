# Beverage Sales Analytics & Forecasting

Individual course project (VNU University of Science, Data Science). Descriptive analysis of a synthetic beverage sales dataset, a Streamlit dashboard, and a basic Prophet revenue forecast with a backtest.

**Dataset:** [Beverage Sales - Kaggle](https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales) by Sebastian Willmann.

The dataset contains synthetic beverage sales data with ~9M order lines,
3M orders, 10K customers, 16 German regions, covering 2021-01-01 to 2023-12-30.

The raw CSV is not included in this repository due to its size.
Download `synthetic_beverage_sales_data.csv` from the Kaggle dataset above
and place it at:

`data/synthetic_beverage_sales_data.csv`

## What is in the analysis
- KPIs: revenue, orders, average order value, customer value, revenue growth
- Revenue trend and seasonality, categories, products, customers (B2B vs B2C), regions
- Discounts: how often they occur and how quantity differs by discount level (descriptive, not causal)
- 12-month revenue forecast with Prophet, **evaluated by a backtest** (train on 2021–2022, test on 2023) against a "same month last year" baseline

## Selected findings (from the notebook)
- Total revenue ≈ 1.18 billion (euro assumed); average order value ≈ 392
- B2B customers generate 76.6% of revenue from 35.6% of orders
- Alcoholic Beverages is the largest category throughout
- The largest customer is a tiny share of revenue, so there is no dependence on a few accounts

See the app and notebook for exact figures, including growth, category shares and the backtest error.

## Run it
```bash
pip install -r requirements.txt

# 1) put the raw CSV at data/synthetic_beverage_sales_data.csv, then build the summary tables (one-off)
python prepare_data.py

# 2) start the dashboard
streamlit run app.py
```
`prepare_data.py` writes small files to `data/processed/`. The app reads only those, so it runs without the raw data. Commit `data/processed/` to deploy the app (for example on Streamlit Community Cloud).

## Files
| File | Purpose |
|---|---|
| `beverage_sales_analysis.ipynb` | Exploratory analysis and forecasting notebook |
| `prepare_data.py` | Builds summary tables and forecasts from the raw CSV |
| `app.py` | Streamlit dashboard |

## Limitations
- Synthetic data: patterns may reflect how the data was generated.
- Three years of history; seasonality and growth estimates are rough.
- No cost, margin, promotion calendar or market-size data.
- Discount analysis is descriptive; it cannot establish cause and effect.
- Forecasts use revenue history only.
