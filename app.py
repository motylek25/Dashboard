import os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# оформление  KPI
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 26px; }
</style>
""", unsafe_allow_html=True)

# форматирование
def fmt_money(val, compact=True):
    val = float(val)
    if compact:
        for factor, suf in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
            if abs(val) >= factor:
                return f"${val/factor:.2f}{suf}"
        return f"${val:,.0f}".replace(",", " ")
    else:
        return f"${val:,.0f}".replace(",", " ")

def fmt_int(val, compact=True):
    val = int(val)
    if compact:
        if abs(val) >= 1e6: return f"{val/1e6:.2f}M"
        if abs(val) >= 1e3: return f"{val/1e3:.1f}K"
    return f"{val:,}".replace(",", " ")

# загрузка с авто-детектом + расширенные алиасы 
@st.cache_data
def load_data(source):
    encodings = ("utf-8", "ISO-8859-1", "latin1")
    seps = (None, ",", ";", "\t")
    df, last_err = None, None
    for enc in encodings:
        for sep in seps:
            try:
                tmp = pd.read_csv(source, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if tmp.shape[1] == 1 and any(ch in tmp.columns[0] for ch in [";", "\t", ","]):
                    continue
                df = tmp
                break
            except Exception as e:
                last_err = e
        if df is not None:
            break
    if df is None:
        raise RuntimeError(f"Не удалось прочитать CSV: {last_err}")

    def canon(s):  # в нижний регистр + убираем не-буквенно-цифровые
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

    # маппинг учитывает варианты без пробелов: productname, orderdate и т.п.
    aliases = {
        "order date":"Order Date", "orderdate":"Order Date",
        "ship date":"Ship Date",   "shipdate":"Ship Date",
        "order id":"Order ID",     "orderid":"Order ID",
        "customer id":"Customer ID","customerid":"Customer ID",
        "customer name":"Customer Name","customername":"Customer Name",
        "segment":"Segment", "country":"Country", "city":"City", "state":"State",
        "postal code":"Postal Code", "postalcode":"Postal Code",
        "region":"Region",
        "product id":"Product ID", "productid":"Product ID",
        "product name":"Product Name", "productname":"Product Name", "product":"Product Name",
        "item name":"Product Name", "itemname":"Product Name",
        "category":"Category",
        "sub category":"Sub-Category", "subcategory":"Sub-Category",
        "sales":"Sales", "quantity":"Quantity", "discount":"Discount", "profit":"Profit"
    }
    df = df.rename(columns={c: aliases[canon(c)] for c in df.columns if canon(c) in aliases})

    # обязательные метрики
    for c in ["Sales", "Profit"]:
        if c not in df.columns:
            raise ValueError(f"Колонка '{c}' обязательна. Найдено: {list(df.columns)}")

    # типы
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df = df.dropna(subset=["Order Date"])
        df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
        df["Year"] = df["Order Date"].dt.year
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    for c in ["Sales","Profit","Discount","Quantity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Profit Margin"] = np.where(df["Sales"] > 0, df["Profit"]/df["Sales"], np.nan)
    return df

# источник 
default_path = "SampleSuperstore.csv"
st.sidebar.title("Superstore Dashboard")
uploaded = st.sidebar.file_uploader("Загрузите CSV (или используйте локальный)", type="csv")
data_src = uploaded if uploaded is not None else (default_path if os.path.exists(default_path) else None)
if not data_src:
    st.warning("Положи рядом с app.py файл 'SampleSuperstore.csv' или загрузите его слева.")
    st.stop()

try:
    df = load_data(data_src)
except Exception as e:
    st.error(f"Ошибка при загрузке: {e}")
    st.stop()

st.sidebar.success(f"Данные загружены: {len(df):,} строк".replace(",", " "))

# фильтры 
compact_kpi = st.sidebar.checkbox("Короткий формат KPI (K/M/B)", value=True)
has_date = "Order Date" in df.columns
if has_date:
    min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
    d1, d2 = st.sidebar.date_input("Order date range", (min_date, max_date),
                                   min_value=min_date, max_value=max_date)
else:
    d1 = d2 = None

vals = lambda col: sorted(df[col].dropna().unique()) if col in df.columns else []
regions, segments, categories = vals("Region"), vals("Segment"), vals("Category")
sel_regions   = st.sidebar.multiselect("Region", regions, default=regions) if regions else []
sel_segments  = st.sidebar.multiselect("Segment", segments, default=segments) if segments else []
sel_categories= st.sidebar.multiselect("Category", categories, default=categories) if categories else []

mask = pd.Series(True, index=df.index)
if has_date:
    mask &= df["Order Date"].dt.date.between(d1, d2)
if regions:
    mask &= df["Region"].isin(sel_regions)
if segments:
    mask &= df["Segment"].isin(sel_segments)
if categories:
    mask &= df["Category"].isin(sel_categories)

dff = df.loc[mask].copy()
if dff.empty:
    st.warning("По выбранным фильтрам данных нет.")
    st.stop()

# KPI (2 строки) 
sales = float(dff["Sales"].sum())
profit = float(dff["Profit"].sum())
items = int(dff["Quantity"].sum()) if "Quantity" in dff.columns else int(len(dff))
avg_discount = float(dff["Discount"].mean()) if "Discount" in dff.columns else 0.0
margin = profit / sales if sales > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Sales",   fmt_money(sales, compact_kpi))
c2.metric("Profit",  fmt_money(profit, compact_kpi))
c3.metric("Items",   fmt_int(items, compact_kpi))

c4, c5 = st.columns(2)
c4.metric("Avg. Discount", f"{avg_discount*100:.1f}%")
c5.metric("Profit margin", f"{margin*100:.1f}%")

# динамика по месяцам (если есть даты) 
if has_date:
    st.subheader("Динамика по месяцам")
    by_month = dff.groupby("Month", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
    by_month["Margin"] = np.where(by_month["Sales"] > 0, by_month["Profit"]/by_month["Sales"], 0.0)

    show_margin = st.checkbox("Показывать маржу в динамике", value=False)
    if show_margin:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=by_month["Month"], y=by_month["Sales"], name="Sales",  mode="lines+markers"))
        fig.add_trace(go.Scatter(x=by_month["Month"], y=by_month["Profit"], name="Profit", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=by_month["Month"], y=by_month["Margin"]*100, name="Margin (%)", mode="lines+markers"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Sales / Profit", secondary_y=False)
        fig.update_yaxes(title_text="Margin, %", secondary_y=True, rangemode="tozero")
        fig.update_layout(template="simple_white", legend_title_text="", xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig1 = px.line(by_month, x="Month", y=["Sales","Profit"], markers=True, template="simple_white")
        fig1.update_layout(legend_title_text="", xaxis_title="", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)

#  категории / Подкатегории 
left, right = st.columns(2)

with left:
    st.subheader("Продажи по категориям")
    if "Category" in dff.columns:
        cat = dff.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        fig2 = px.bar(cat, x="Category", y="Sales", text_auto=".2s", template="simple_white")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Колонка 'Category' отсутствует в файле.")

with right:
    st.subheader("Подкатегории (топ‑10 по Sales)")
    if "Sub-Category" in dff.columns:
        sub = dff.groupby("Sub-Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
        fig3 = px.bar(sub, x="Sales", y="Sub-Category", orientation="h", text_auto=".2s", template="simple_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Колонка 'Sub-Category' отсутствует в файле.")

# топ по прибыли 
st.subheader("Топ‑10 по прибыли")
# если есть 'Product Name' — используем, иначе fallback на Sub-Category/Category
dim = "Product Name" if "Product Name" in dff.columns else ("Sub-Category" if "Sub-Category" in dff.columns else ("Category" if "Category" in dff.columns else None))
if dim is not None:
    top = dff.groupby(dim, as_index=False).agg(Profit=("Profit","sum"), Sales=("Sales","sum"))
    top = top.sort_values("Profit", ascending=False).head(10)
    fig4 = px.bar(top, x="Profit", y=dim, orientation="h", text_auto=".2s",
                  template="simple_white", hover_data=["Sales"])
    fig4.update_layout(xaxis_title="Profit", yaxis_title=dim)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Нет подходящего признака для топа (ожидаются 'Product Name' или 'Sub-Category' / 'Category').")

# штаты 
st.subheader("Топ‑10 штатов по продажам")
if "State" in dff.columns:
    by_state = dff.groupby("State", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
    by_state = by_state.sort_values("Sales", ascending=False).head(10)
    fig5 = px.bar(by_state, x="Sales", y="State", orientation="h", text_auto=".2s",
                  template="simple_white", color="Profit", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Колонка 'State' отсутствует в файле.")

# scatter скидка‑прибыль 
with st.expander("Discount vs Profit (scatter)"):
    if {"Discount","Profit"}.issubset(dff.columns):
        sample = dff.sample(min(5000, len(dff)), random_state=42)
        color_col = "Category" if "Category" in dff.columns else None
        fig6 = px.scatter(sample, x="Discount", y="Profit", color=color_col,
                          opacity=0.6, template="simple_white")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Нужны колонки Discount и Profit.")

#  выгрузка 
csv = dff.to_csv(index=False).encode("utf-8")
st.download_button("Скачать отфильтрованные данные (CSV)",
                   data=csv, file_name="superstore_filtered.csv", mime="text/csv")

st.caption("Источник: Kaggle — SampleSuperstore. Приложение устойчиво к разным именам колонок и отсутствию Order Date.")