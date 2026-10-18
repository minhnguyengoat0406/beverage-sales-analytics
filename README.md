# Beverage Sales Analytics & Forecasting

Individual Data Science project focused on exploratory sales analysis, customer and product insights, and revenue forecasting using Python.

The project analyzes a large synthetic beverage sales dataset and includes a Jupyter Notebook for the analysis, a Streamlit dashboard, and a data preparation pipeline for generating the aggregated data used by the dashboard.

## Dataset

**Source:** [Beverage Sales - Kaggle](https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales) by Sebastian Willmann.

The dataset contains synthetic beverage sales data with approximately:

- 9 million order lines
- 3 million orders
- 10,000 customers
- 16 German regions
- Data from 2021-01-01 to 2023-12-30

The raw CSV is not included in this repository due to its size.

To reproduce the project, download `synthetic_beverage_sales_data.csv` from the Kaggle dataset above and place it at:

```text
data/synthetic_beverage_sales_data.csv
```

## What is in the analysis

The analysis covers:

- Revenue, orders, customers, and average order value
- Revenue trends and potential seasonality
- Category and product performance
- B2B vs B2C customer behavior
- Customer revenue concentration
- Regional sales performance
- Discount patterns and order quantities
- 12-month revenue forecasting using Prophet
- Forecast evaluation through backtesting
- Comparison with a simple seasonal-naive baseline

The discount analysis is descriptive rather than causal: the available data does not allow the effect of discounts on sales to be isolated from customer type and order size.

## Selected Findings

Some of the main findings from the analysis:

- Total revenue is approximately €1.18 billion.
- Average order value is approximately €392.
- B2B customers generate 76.6% of revenue while accounting for 35.6% of orders.
- Alcoholic Beverages is the largest product category throughout the dataset.
- Revenue is distributed across a large customer base rather than being highly dependent on a few major customers.
- The forecasting model is evaluated against a "same month last year" baseline rather than relying only on in-sample fit.

See `beverage_sales_analysis.ipynb` for the complete analysis, visualizations, and exact results.

## Project Structure

| File | Purpose |
|---|---|
| `beverage_sales_analysis.ipynb` | Exploratory data analysis, visualizations, insights, and forecasting |
| `prepare_data.py` | Processes the raw dataset and generates aggregated tables and forecasts |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies required by the project |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `synthetic_beverage_sales_data.csv` from the Kaggle dataset linked above.

Create a `data` directory and place the file at:

```text
data/synthetic_beverage_sales_data.csv
```

### 3. Prepare the data

Run:

```bash
python prepare_data.py
```

The script processes the raw dataset and generates smaller aggregated tables and forecasting outputs in:

```text
data/processed/
```

These generated files are not included in this repository.

### 4. Start the dashboard

After the processed data has been generated, run:

```bash
streamlit run app.py
```

The Streamlit application reads the aggregated files from `data/processed/` rather than loading the full raw dataset directly.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Plotly
- Prophet
- Streamlit
- Jupyter Notebook

## Limitations

- The dataset is synthetic, so observed patterns may reflect how the data was generated rather than real consumer behavior.
- Only three years of historical data are available, making long-term growth and seasonality estimates uncertain.
- The dataset does not contain cost, margin, promotion-calendar, or market-size information.
- Discount analysis is descriptive and cannot establish a causal relationship between discounts and sales.
- Revenue forecasts use historical revenue patterns only and do not incorporate external economic or market variables.