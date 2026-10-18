"""
Build the small summary tables used by the Streamlit app.

The raw dataset (~9M rows) is too large for GitHub or a free hosting tier, so this
script is run once locally. It reads the raw CSV, computes every aggregate and the
Prophet forecasts, and writes small files to data/processed/. The app only reads those.

Usage:
    python prepare_data.py
    python prepare_data.py --input path/to/synthetic_beverage_sales_data.csv
    python prepare_data.py --skip-forecast        # aggregates only (no Prophet needed)
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["Order_ID", "Customer_ID", "Customer_Type", "Product", "Category",
        "Unit_Price", "Quantity", "Discount", "Total_Price", "Region", "Order_Date"]

SPLIT = "2023-01-01"      # backtest: train on data before this date, test on the 12 months after
HORIZON = 12              # months to forecast
FOURIER_ORDER = 3         # yearly seasonality complexity (default 10 is too flexible for 36 points)


# --------------------------------------------------------------------------- aggregates
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, usecols=COLS, parse_dates=["Order_Date"],
        dtype={"Customer_Type": "category", "Category": "category",
               "Region": "category", "Product": "category"},
    )
    df["Month"] = df["Order_Date"].dt.to_period("M").dt.to_timestamp()
    df["Discount"] = df["Discount"].round(2)
    return df


def build_aggregates(df: pd.DataFrame, out: Path) -> dict:
    total_revenue = float(df["Total_Price"].sum())
    total_orders = int(df["Order_ID"].nunique())

    # --- by customer type
    by_type = df.groupby("Customer_Type", observed=True).agg(
        revenue=("Total_Price", "sum"), orders=("Order_ID", "nunique")).reset_index()
    cust_type = df.groupby("Customer_ID", observed=True)["Customer_Type"].first()
    by_type["customers"] = by_type["Customer_Type"].map(cust_type.value_counts()).astype(int)
    by_type["aov"] = by_type["revenue"] / by_type["orders"]
    by_type.to_csv(out / "customer_type.csv", index=False)

    # --- customers
    cust_rev = df.groupby("Customer_ID", observed=True)["Total_Price"].sum().sort_values(ascending=False)
    n_cust = len(cust_rev)
    top_n = max(1, int(n_cust * 0.10))
    cust_orders = df.groupby("Customer_ID", observed=True)["Order_ID"].nunique()
    top = cust_rev.head(20).rename("revenue").reset_index()
    top["Customer_Type"] = top["Customer_ID"].map(cust_type).astype(str)
    top["orders"] = top["Customer_ID"].map(cust_orders)
    top.to_csv(out / "top_customers.csv", index=False)

    # --- monthly
    g = df.groupby("Month", observed=True)
    monthly = pd.DataFrame({
        "revenue": g["Total_Price"].sum(),
        "orders": g["Order_ID"].nunique(),
        "days_with_data": g["Order_Date"].nunique(),
    }).reset_index()
    monthly["revenue_per_day"] = monthly["revenue"] / monthly["days_with_data"]
    monthly.to_csv(out / "monthly.csv", index=False)

    df.groupby(["Month", "Category"], observed=True)["Total_Price"].sum() \
        .rename("revenue").reset_index().to_csv(out / "category_monthly.csv", index=False)

    df.groupby(["Product", "Category"], observed=True).agg(
        revenue=("Total_Price", "sum"), quantity=("Quantity", "sum")) \
        .reset_index().sort_values("revenue", ascending=False).to_csv(out / "products.csv", index=False)

    df.groupby("Region", observed=True).agg(
        revenue=("Total_Price", "sum"), orders=("Order_ID", "nunique")) \
        .reset_index().sort_values("revenue", ascending=False).to_csv(out / "regions.csv", index=False)

    # --- discounts (one row = one order line)
    df.groupby(["Customer_Type", "Discount"], observed=True).agg(
        lines=("Order_ID", "size"), avg_quantity=("Quantity", "mean"),
        avg_unit_price=("Unit_Price", "mean"), revenue=("Total_Price", "sum")) \
        .reset_index().to_csv(out / "discount_levels.csv", index=False)

    df.assign(discounted=df["Discount"] > 0).groupby("Category", observed=True).agg(
        lines=("Order_ID", "size"), discounted_share=("discounted", "mean"),
        avg_discount=("Discount", "mean")) \
        .reset_index().to_csv(out / "discount_category.csv", index=False)

    kpis = {
        "rows": int(len(df)),
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": int(n_cust),
        "regions": int(df["Region"].nunique()),
        "aov": total_revenue / total_orders,
        "date_min": str(df["Order_Date"].min().date()),
        "date_max": str(df["Order_Date"].max().date()),
        "top_customer_share": float(cust_rev.iloc[0] / total_revenue),
        "top_10pct_customer_share": float(cust_rev.head(top_n).sum() / total_revenue),
    }
    (out / "kpis.json").write_text(json.dumps(kpis, indent=2))
    return kpis


# --------------------------------------------------------------------------- forecasting
def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def make_model():
    from prophet import Prophet
    return Prophet(yearly_seasonality=FOURIER_ORDER, weekly_seasonality=False,
                   daily_seasonality=False)


def run_forecast(series: pd.DataFrame):
    """series: columns ds (month start), y. Returns (forecast, backtest, metrics)."""
    series = series.sort_values("ds").reset_index(drop=True)
    train, test = series[series["ds"] < SPLIT], series[series["ds"] >= SPLIT]

    backtest, metrics = None, {}
    if len(train) >= 24 and len(test) > 0:
        m = make_model().fit(train)
        pred = m.predict(pd.DataFrame({"ds": test["ds"]}))
        lag = series.set_index("ds")["y"].reindex(test["ds"] - pd.DateOffset(months=12)).values
        backtest = pd.DataFrame({"ds": test["ds"].values, "actual": test["y"].values,
                                 "prophet": pred["yhat"].values, "seasonal_naive": lag})
        metrics = {"mape_prophet": mape(backtest["actual"], backtest["prophet"]),
                   "mape_seasonal_naive": mape(backtest["actual"], backtest["seasonal_naive"]),
                   "test_months": int(len(test))}

    m = make_model().fit(series)
    fc = m.predict(m.make_future_dataframe(periods=HORIZON, freq="MS"))
    fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc["kind"] = np.where(fc["ds"] > series["ds"].max(), "forecast", "fit")
    return fc, backtest, metrics


def build_forecasts(out: Path):
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    monthly = pd.read_csv(out / "monthly.csv", parse_dates=["Month"])
    fc, bt, metrics = run_forecast(monthly.rename(columns={"Month": "ds", "revenue": "y"})[["ds", "y"]])
    fc.to_csv(out / "forecast_total.csv", index=False)
    if bt is not None:
        bt.to_csv(out / "backtest_total.csv", index=False)
    (out / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))

    cat = pd.read_csv(out / "category_monthly.csv", parse_dates=["Month"])
    fcs, rows = [], []
    for name, sub in cat.groupby("Category"):
        sub = sub.rename(columns={"Month": "ds", "revenue": "y"})[["ds", "y"]]
        sub = sub[sub["y"] > 0]
        if len(sub) < 24:
            continue
        f, _, mt = run_forecast(sub)
        f["Category"] = name
        fcs.append(f)
        rows.append({"Category": name, **mt})
    pd.concat(fcs).to_csv(out / "forecast_category.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "backtest_category.csv", index=False)


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/synthetic_beverage_sales_data.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--skip-forecast", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Loading raw data ...")
    kpis = build_aggregates(load_raw(Path(a.input)), out)
    print(f"Aggregates written to {out}  ({kpis['rows']:,} rows)")
    if not a.skip_forecast:
        print("Fitting Prophet models ...")
        build_forecasts(out)
        print("Forecasts written.")
