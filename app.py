"""
Beverage Sales Analytics & Forecasting Dashboard.

Reads only the small summary tables in data/processed/ (created by prepare_data.py).
Every number in the commentary is computed from those tables, not typed by hand.
"""
import calendar
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data" / "processed"

st.set_page_config(page_title="Beverage Sales Analytics", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data
def load():
    d = {"kpis": json.loads((DATA / "kpis.json").read_text())}
    d["monthly"] = pd.read_csv(DATA / "monthly.csv", parse_dates=["Month"])
    d["cat_monthly"] = pd.read_csv(DATA / "category_monthly.csv", parse_dates=["Month"])
    for k, f in [("products", "products"), ("regions", "regions"), ("ctype", "customer_type"),
                 ("top_cust", "top_customers"), ("disc_levels", "discount_levels"),
                 ("disc_cat", "discount_category")]:
        d[k] = pd.read_csv(DATA / f"{f}.csv")
    if (DATA / "forecast_total.csv").exists():
        d["fc"] = pd.read_csv(DATA / "forecast_total.csv", parse_dates=["ds"])
        d["metrics"] = json.loads((DATA / "backtest_metrics.json").read_text())
        d["bt"] = (pd.read_csv(DATA / "backtest_total.csv", parse_dates=["ds"])
                   if (DATA / "backtest_total.csv").exists() else None)
        d["fc_cat"] = pd.read_csv(DATA / "forecast_category.csv", parse_dates=["ds"])
        d["bt_cat"] = pd.read_csv(DATA / "backtest_category.csv")
    return d


if not (DATA / "kpis.json").exists():
    st.error("Processed data not found. Run `python prepare_data.py` first (see README).")
    st.stop()

D = load()
K = D["kpis"]


def eur(x, m=False):
    return f"€{x / 1e6:,.1f}M" if m else f"€{x:,.0f}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def note(text):
    st.info(text, icon="🔎")


# --------------------------------------------------------------------------- header + KPIs
st.title("Beverage Sales Analytics & Forecasting")
st.caption(
    f"Synthetic dataset · {K['rows']:,} order lines · {K['date_min']} to {K['date_max']} · "
    "currency assumed to be euro (regions are German states)."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total revenue", eur(K["total_revenue"], m=True))
c2.metric("Orders", f"{K['total_orders']:,}")
c3.metric("Customers", f"{K['total_customers']:,}")
c4.metric("Average order value", eur(K["aov"]))
c5.metric("Regions", K["regions"])

st.divider()

# --------------------------------------------------------------------------- revenue trend
st.header("Revenue trend")
m = D["monthly"].copy()
m["Year"], m["MonthNum"] = m["Month"].dt.year, m["Month"].dt.month

view = st.radio("View", ["Monthly total", "Average per day"], horizontal=True)
ycol = "revenue" if view == "Monthly total" else "revenue_per_day"
fig = px.line(m, x="Month", y=ycol, markers=True, labels={ycol: "Revenue (€)", "Month": ""})
st.plotly_chart(fig)

yearly = m.groupby("Year")["revenue"].sum()
years = list(yearly.index)
growth = [f"{years[i]} vs {years[i-1]}: {(yearly.iloc[i] / yearly.iloc[i-1] - 1) * 100:+.1f}%"
          for i in range(1, len(years))]

m["idx"] = m["revenue_per_day"] / m.groupby("Year")["revenue_per_day"].transform("mean")
season = m.groupby("MonthNum")["idx"].mean()
hi, lo = season.idxmax(), season.idxmin()

note(
    f"Yearly revenue growth ({', '.join(growth)}). The last year covers data up to {K['date_max']}.  \n"
    f"Monthly totals partly reflect month length (February is shorter), so use the *Average per day* view "
    f"to compare months. Averaged over the years, revenue per day was highest in {calendar.month_abbr[hi]} "
    f"(index {season[hi]:.2f}) and lowest in {calendar.month_abbr[lo]} ({season[lo]:.2f}), where 1.00 is the "
    f"yearly average. With only {len(years)} years of data this is a hint of seasonality, not proof."
)

st.divider()

# --------------------------------------------------------------------------- categories & products
st.header("Categories and products")
cat = D["cat_monthly"]
fig = px.area(cat, x="Month", y="revenue", color="Category", labels={"revenue": "Revenue (€)", "Month": ""})
st.plotly_chart(fig)

cat_tot = cat.groupby("Category")["revenue"].sum().sort_values(ascending=False)
top_cat = cat_tot.index[0]
cat["Year"] = cat["Month"].dt.year
yr_tot = cat.groupby("Year")["revenue"].sum()
top_by_year = cat[cat["Category"] == top_cat].groupby("Year")["revenue"].sum() / yr_tot
note(
    f"{top_cat} accounts for {pct(cat_tot.iloc[0] / cat_tot.sum())} of total revenue "
    f"({pct(top_by_year.min())}–{pct(top_by_year.max())} depending on the year). "
    "Other categories: " + ", ".join(f"{c} {pct(v / cat_tot.sum())}" for c, v in cat_tot.iloc[1:].items()) + "."
)

prod = D["products"]
top10 = prod.head(10)
fig = px.bar(top10, x="Product", y="revenue", color="Category",
             labels={"revenue": "Revenue (€)", "Product": ""}, title="Top 10 products by revenue")
st.plotly_chart(fig)
note(
    f"The top 10 products (out of {len(prod)}) generate {pct(top10['revenue'].sum() / prod['revenue'].sum())} "
    f"of revenue; the best seller, {top10.iloc[0]['Product']}, alone accounts for "
    f"{pct(top10.iloc[0]['revenue'] / prod['revenue'].sum())}. Revenue ranking reflects price as well as volume: "
    "premium items can rank high with fewer units sold."
)

st.divider()

# --------------------------------------------------------------------------- customers
st.header("Customers")
ct = D["ctype"].copy()
ct["revenue_share"] = ct["revenue"] / ct["revenue"].sum()
ct["order_share"] = ct["orders"] / ct["orders"].sum()
ct["revenue_per_customer"] = ct["revenue"] / ct["customers"]

a, b = st.columns(2)
a.plotly_chart(px.pie(ct, names="Customer_Type", values="revenue", title="Revenue share"))
b.plotly_chart(px.pie(ct, names="Customer_Type", values="orders", title="Order share"))

show = ct[["Customer_Type", "customers", "orders", "revenue", "aov", "revenue_per_customer"]].rename(columns={
    "Customer_Type": "Type", "customers": "Customers", "orders": "Orders", "revenue": "Revenue (€)",
    "aov": "AOV (€)", "revenue_per_customer": "Revenue per customer (€)"})
st.dataframe(show.style.format({"Customers": "{:,.0f}", "Orders": "{:,.0f}", "Revenue (€)": "{:,.0f}",
                                "AOV (€)": "{:,.0f}", "Revenue per customer (€)": "{:,.0f}"}),
             hide_index=True)

r = ct.set_index("Customer_Type")
if {"B2B", "B2C"} <= set(r.index):
    note(
        f"B2B generates {pct(r.loc['B2B', 'revenue_share'])} of revenue from {pct(r.loc['B2B', 'order_share'])} "
        f"of orders. Its average order is {r.loc['B2B', 'aov'] / r.loc['B2C', 'aov']:.1f}× the B2C average "
        f"({eur(r.loc['B2B', 'aov'])} vs {eur(r.loc['B2C', 'aov'])})."
        + (" Larger B2B orders lift the overall AOV." if r.loc['B2B', 'aov'] > 1.2 * r.loc['B2C', 'aov'] else "")
    )

top_c = D["top_cust"].head(10)
fig = px.bar(top_c, x="Customer_ID", y="revenue", color="Customer_Type",
             labels={"revenue": "Revenue (€)", "Customer_ID": ""}, title="Top 10 customers by revenue")
fig.update_xaxes(type="category")
st.plotly_chart(fig)
note(
    f"Revenue is spread across {K['total_customers']:,} customers: the largest customer represents only "
    f"{pct(K['top_customer_share'], 2)} of revenue and the top 10% of customers represent "
    f"{pct(K['top_10pct_customer_share'])} (a perfectly even split would give 10%)."
)

st.divider()

# --------------------------------------------------------------------------- regions
st.header("Regions")
reg = D["regions"].sort_values("revenue", ascending=False)
fig = px.bar(reg, x="Region", y="revenue", labels={"revenue": "Revenue (€)", "Region": ""})
st.plotly_chart(fig)
hi_r, lo_r = reg.iloc[0], reg.iloc[-1]
note(
    f"Revenue ranges from {eur(lo_r['revenue'], m=True)} ({lo_r['Region']}) to {eur(hi_r['revenue'], m=True)} "
    f"({hi_r['Region']}); the highest region is {hi_r['revenue'] / lo_r['revenue']:.2f}× the lowest and "
    f"{(hi_r['revenue'] / reg['revenue'].mean() - 1) * 100:+.0f}% versus the average. "
    + ("Revenue is fairly evenly spread. " if hi_r['revenue'] / lo_r['revenue'] < 1.3 else "")
    + "Without population or market-size data, these figures show size, not market penetration."
)

st.divider()

# --------------------------------------------------------------------------- discounts
st.header("Discounts")
dl = D["disc_levels"].copy()
dl["discounted"] = dl["Discount"] > 0
dl["wq"] = dl["avg_quantity"] * dl["lines"]

disc_share = dl.loc[dl["discounted"], "lines"].sum() / dl["lines"].sum()
tot = dl.groupby("Customer_Type")["lines"].sum()
dis = dl[dl["discounted"]].groupby("Customer_Type")["lines"].sum().reindex(tot.index).fillna(0)
type_share = dis / tot

dl["Discount level"] = (dl["Discount"] * 100).round().astype(int).astype(str) + "%"
fig = px.bar(dl, x="Discount level", y="avg_quantity", color="Customer_Type", barmode="group",
             labels={"avg_quantity": "Average quantity per order line"},
             title="Average quantity per order line, by discount level and customer type")
st.plotly_chart(fig)

lvl = dl[dl["discounted"]].groupby("Discount")[["lines", "wq"]].sum()
lvl["avg_quantity"] = lvl["wq"] / lvl["lines"]
all_or_nothing = bool(((type_share == 0) | (type_share == 1)).all())
rising = bool(lvl["avg_quantity"].is_monotonic_increasing) and len(lvl) > 1

text = (f"{pct(disc_share)} of order lines carry a discount. Share of lines discounted by customer type: "
        + ", ".join(f"{t} {pct(v)}" for t, v in type_share.items()) + ".  \n")
if all_or_nothing:
    text += "Whether a line is discounted is decided entirely by customer type. "
if rising:
    text += ("Among discounted lines, average quantity rises with the discount level ("
             + ", ".join(f"{int(round(d * 100))}%: {q:.0f} units" for d, q in lvl["avg_quantity"].items())
             + "), which looks like a volume-tier scheme.  \n")
text += ("**The data cannot show whether discounts increase sales.** Customer type and order size move together "
         "with the discount, so any difference in quantity or revenue could be due to them rather than the discount. "
         "A causal answer would need promotion timing or comparable customers with and without a discount.")
note(text)

st.divider()

# --------------------------------------------------------------------------- forecast
st.header("Revenue forecast")
if "fc" not in D:
    st.warning("Forecast files not found. Run `python prepare_data.py` without `--skip-forecast`.")
else:
    fc, mt = D["fc"], D["metrics"]
    hist = D["monthly"]
    fut = fc[fc["kind"] == "forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat_upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat_lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(99,110,250,0.2)", name="80% interval"))
    fig.add_trace(go.Scatter(x=hist["Month"], y=hist["revenue"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], mode="lines+markers", name="Forecast",
                             line=dict(dash="dash")))
    fig.update_layout(title=f"{len(fut)}-month revenue forecast (Prophet)", yaxis_title="Revenue (€)")
    st.plotly_chart(fig)

    last12 = hist.sort_values("Month").tail(len(fut))["revenue"].sum()
    st.markdown(
        f"The forecast totals {eur(fut['yhat'].sum(), m=True)} for the next {len(fut)} months, "
        f"{(fut['yhat'].sum() / last12 - 1) * 100:+.1f}% versus the last {len(fut)} months of actual data."
    )

    st.subheader("How reliable is it? Backtest on 2023")
    if D.get("bt") is not None:
        bt = D["bt"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["ds"], y=bt["actual"], name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=bt["ds"], y=bt["prophet"], name="Prophet", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=bt["ds"], y=bt["seasonal_naive"], name="Same month last year",
                                 mode="lines+markers", line=dict(dash="dot")))
        fig.update_layout(yaxis_title="Revenue (€)",
                          title="Model trained on 2021–2022, tested on 2023")
        st.plotly_chart(fig)
        better = mt["mape_prophet"] < mt["mape_seasonal_naive"]
        note(
            f"Prophet's error on the 2023 test months (MAPE) is {mt['mape_prophet']:.1f}%, compared with "
            f"{mt['mape_seasonal_naive']:.1f}% for the simple rule “same month as last year”. "
            + ("Prophet is more accurate than this baseline." if better else
               "Prophet does **not** beat this simple baseline, so treat the forecast with caution.")
            + " With only 24 training months, the estimate of seasonality is rough."
        )

    fcc = D["fc_cat"]
    fig = px.line(fcc[fcc["kind"] == "forecast"], x="ds", y="yhat", color="Category",
                  labels={"yhat": "Forecast revenue (€)", "ds": ""}, title="Forecast by category")
    st.plotly_chart(fig)
    bc = D["bt_cat"].copy()
    if len(bc):
        st.dataframe(bc.rename(columns={"mape_prophet": "Prophet MAPE (%)",
                                        "mape_seasonal_naive": "Baseline MAPE (%)",
                                        "test_months": "Test months"})
                     .style.format({"Prophet MAPE (%)": "{:.1f}", "Baseline MAPE (%)": "{:.1f}"}),
                     hide_index=True)
    st.caption("Category forecasts come from separate models; the category sums are not forced to equal the total forecast.")

st.divider()

# --------------------------------------------------------------------------- limitations
st.header("Limitations")
st.markdown(
    """
- The data is **synthetic**, so patterns (for example the regional split) may reflect how it was generated, not real behaviour.
- Only **three years** of history: seasonality and growth estimates are rough.
- No cost, margin, promotion calendar, or market-size data, so profitability, promotion effect and market penetration cannot be assessed.
- Forecasts use revenue history only (no external factors) and carry real uncertainty; see the backtest above.
"""
)
st.caption("Beverage Sales Analytics & Forecasting · Nguyen Duc Minh")
