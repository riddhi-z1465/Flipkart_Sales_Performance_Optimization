import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Flipkart Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Flipkart Sales Analytics\nPowered by Machine Learning"
    }
)

# ── Stable dashboard theme ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 10%, rgba(88,192,218,0.14), transparent 36%),
        radial-gradient(circle at 92% 14%, rgba(240,154,70,0.12), transparent 35%),
        linear-gradient(145deg, #0b1220 0%, #111c2e 50%, #1b2a3f 100%);
}

[data-testid="stAppViewContainer"] * {
    font-family: 'Manrope', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c2d 0%, #1a2a3f 100%);
    border-right: 1px solid rgba(88,192,218,0.25);
}

[data-testid="metric-container"] {
    background: rgba(19, 32, 50, 0.84);
    border: 1px solid rgba(88, 192, 218, 0.25);
    border-radius: 16px;
    padding: 18px 22px;
}

.stTabs [data-baseweb="tab-list"] {
    border: 1px solid rgba(88,192,218,0.25);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2f7f9b, #58c0da);
    color: #fff;
    border-radius: 8px;
}

.stButton > button {
    background: linear-gradient(135deg, #2f7f9b, #58c0da);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(130deg, #7ad4eb 0%, #58c0da 52%, #f09a46 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    color: #9fd8e8;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #58c0da, #f09a46) 1;
    padding-bottom: 0.5rem;
    margin: 1.8rem 0 1.1rem 0;
}

.info-box, .success-box, .warning-box, .danger-box {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.info-box {
    background: rgba(88,192,218,0.12);
    border-left: 4px solid #58c0da;
    color: #b9e2ef;
}

.success-box {
    background: rgba(52,211,153,0.10);
    border-left: 4px solid #34d399;
    color: #a7f3d0;
}

.warning-box {
    background: rgba(240,154,70,0.10);
    border-left: 4px solid #f09a46;
    color: #ffcc96;
}

.danger-box {
    background: rgba(239,68,68,0.10);
    border-left: 4px solid #ef4444;
    color: #fca5a5;
}

.sidebar-brand {
    text-align: center;
    padding: 1.2rem 0 0.6rem 0;
}

.sidebar-brand .logo-text {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #58c0da, #f09a46);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand .logo-sub {
    font-size: 0.72rem;
    color: #9bb0c5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.feature-pill {
    display: inline-block;
    background: rgba(88,192,218,0.12);
    border: 1px solid rgba(88,192,218,0.25);
    color: #a9dceb;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.73rem;
    font-weight: 600;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def resolve_csv_path():
    """Pick an available CSV source, preferring full dataset."""
    full_csv = Path("Sales.csv")
    sample_csv = Path("Sales_sample.csv")
    if full_csv.exists():
        return full_csv
    if sample_csv.exists():
        return sample_csv
    return None


@st.cache_data
def load_data(nrows=None):
    """Load only required columns with compact dtypes for faster IO."""
    csv_path = resolve_csv_path()
    if csv_path is None:
        raise FileNotFoundError(
            "No CSV dataset found. Add Sales.csv or Sales_sample.csv to the project root, or provide sales_preprocessed.pkl."
        )

    use_cols = [
        "dim_customer_key",
        "procured_quantity",
        "unit_selling_price",
        "total_discount_amount",
        "total_weighted_landing_price",
        "order_id",
        "date_",
        "city_name",
    ]
    dtype_map = {
        "dim_customer_key": "int32",
        "procured_quantity": "int16",
        "unit_selling_price": "float32",
        "total_discount_amount": "float32",
        "total_weighted_landing_price": "float32",
    }
    return pd.read_csv(
        csv_path,
        usecols=use_cols,
        dtype=dtype_map,
        parse_dates=["date_"],
        infer_datetime_format=True,
        nrows=nrows,
    )

@st.cache_data
def prepare_customer_df(df):
    customer_df = (
        df.groupby("dim_customer_key")
        .agg({
            "Quantity": "sum",
            "Revenue": "sum",
            "Discount": "sum",
            "order_id": "nunique",
        })
        .reset_index()
        .rename(columns={"order_id": "Total_Orders"})
    )
    return customer_df

@st.cache_resource
def train_regression(df):
    # Sampling keeps startup responsive on very large datasets.
    if len(df) > 300_000:
        reg_df = df.sample(n=300_000, random_state=42)
    else:
        reg_df = df

    X = reg_df[["Quantity", "Discount", "Month"]]
    y = reg_df["Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))

@st.cache_data
def sample_data(df, n=5000):
    """Sample data for visualization to reduce message size"""
    if len(df) > n:
        return df.sample(n=n, random_state=42)
    return df


def top_values(series, n=300):
    """Return top-N frequent values to avoid huge widget payloads."""
    return series.value_counts().head(n).index.tolist()

# Load data only once and cache it
@st.cache_data
def preprocess_data(df):
    """Preprocess data once and cache the result"""
    # If called with already-preprocessed data from disk cache, return as-is.
    if {"Revenue", "Month", "Year", "Day", "Quantity", "Selling_Price", "Discount", "Landing_Price"}.issubset(df.columns):
        return df

    df["Year"] = df["date_"].dt.year
    df["Month"] = df["date_"].dt.month
    df["Day"] = df["date_"].dt.day
    df.rename(
        columns={
            "procured_quantity": "Quantity",
            "unit_selling_price": "Selling_Price",
            "total_discount_amount": "Discount",
            "total_weighted_landing_price": "Landing_Price",
        },
        inplace=True,
    )
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    # Downcast derived columns as well.
    df["Revenue"] = pd.to_numeric(df["Revenue"], downcast="float")
    df["Month"] = pd.to_numeric(df["Month"], downcast="integer")
    df["Day"] = pd.to_numeric(df["Day"], downcast="integer")
    df["Year"] = pd.to_numeric(df["Year"], downcast="integer")
    return df

@st.cache_data
def get_or_build_dataset():
    """Persist preprocessed dataframe to disk for fast app restarts."""
    csv_path = resolve_csv_path()
    cache_path = Path("sales_preprocessed.pkl")

    if cache_path.exists():
        # If raw CSV is unavailable (common on cloud deploy), serve from cached pickle.
        if (csv_path is None) or cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                pass

    if csv_path is None:
        raise FileNotFoundError(
            "Missing dataset. Add Sales.csv or Sales_sample.csv to the app root, or ship sales_preprocessed.pkl."
        )

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)
    processed_df.to_pickle(cache_path)
    return processed_df

@st.cache_data
def get_or_build_overview_snapshot():
    """Create a compact snapshot so Overview can render without loading full dataset."""
    snapshot_path = Path("overview_snapshot.pkl")
    data_path = Path("sales_preprocessed.pkl")

    if snapshot_path.exists() and data_path.exists():
        if snapshot_path.stat().st_mtime >= data_path.stat().st_mtime:
            try:
                return pd.read_pickle(snapshot_path)
            except Exception:
                pass

    df_full = get_or_build_dataset()

    top_cities_revenue = (
        df_full.groupby("city_name")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_cities_orders = (
        df_full.groupby("city_name")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Keep histogram payload small while preserving visual fidelity.
    hist_sample_size = min(25_000, len(df_full))
    hist_df = df_full.sample(n=hist_sample_size, random_state=42) if len(df_full) > hist_sample_size else df_full

    snapshot = {
        "kpis": {
            "total_revenue": float(df_full["Revenue"].sum()),
            "total_orders": int(df_full["order_id"].nunique()),
            "total_units": int(df_full["Quantity"].sum()),
            "total_customers": int(df_full["dim_customer_key"].nunique()),
            "avg_discount": float(df_full["Discount"].mean()),
            "avg_selling_price": float(df_full["Selling_Price"].mean()),
            "row_count": int(len(df_full)),
        },
        "monthly_revenue": df_full.groupby("Month")["Revenue"].sum().reset_index(),
        "monthly_quantity": df_full.groupby("Month")["Quantity"].sum().reset_index(),
        "hist_sample": hist_df[["Revenue", "Quantity", "Discount"]],
        "top_cities_revenue": top_cities_revenue,
        "top_cities_orders": top_cities_orders,
        "preview": df_full.head(200),
    }

    pd.to_pickle(snapshot, snapshot_path)
    return snapshot


def show_missing_data_help(error):
    st.error(f"Dataset not found: {error}")
    st.markdown(
        """
        ### How to fix
        1. Add `Sales_sample.csv` (recommended for cloud deploy) to the repository root.
        2. Or add full `Sales.csv` to the repository root if your hosting allows it.
        3. Or upload a prebuilt `sales_preprocessed.pkl` to the repository root.
        4. Reboot the Streamlit app after files are available.
        """
    )
    st.stop()


def get_df():
    """Lazy-load full dataset only when a page needs detailed analysis."""
    if "df_cache" not in st.session_state:
        try:
            with st.spinner("Loading full dataset..."):
                st.session_state["df_cache"] = get_or_build_dataset()
        except FileNotFoundError as error:
            show_missing_data_help(error)
    return st.session_state["df_cache"]


def get_overview():
    """Load overview snapshot with graceful handling when dataset files are missing."""
    try:
        return get_or_build_overview_snapshot()
    except FileNotFoundError as error:
        show_missing_data_help(error)


def get_model_metrics():
    """Lazy-train model only when Prediction page is opened."""
    if "model_cache" not in st.session_state:
        df_for_model = get_df()
        with st.spinner("Training prediction model..."):
            st.session_state["model_cache"] = train_regression(df_for_model)
    return st.session_state["model_cache"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo-text">FlipAnalytics</div>
        <div class="logo-sub">Sales Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["📊 Overview", "📈 EDA", "👥 Clustering", "🔮 Prediction"],
    label_visibility="collapsed"
)

with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="margin-bottom:0.5rem; font-size:0.78rem; font-weight:700; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">Features</div>
    <span class="feature-pill">📈 Sales Analytics</span>
    <span class="feature-pill">👥 Segmentation</span>
    <span class="feature-pill">🔮 Forecasting</span>
    <span class="feature-pill">💹 Profit Analysis</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"💾 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Powered by Streamlit + Plotly")

# ── Dark Plotly layout helper ─────────────────────────────────────────────────
_DARK_BG   = "rgba(10,19,31,0)"
_PAPER_BG  = "rgba(16,30,47,0.88)"
_GRID_CLR  = "rgba(88,192,218,0.12)"
_FONT_CLR  = "#9bb0c5"
_TITLE_CLR = "#e8f1f8"

def dark_layout(fig, title_size=15, height=None, **kwargs):
    """Apply consistent dark theme to a Plotly figure."""
    upd = dict(
        plot_bgcolor=_DARK_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(color=_FONT_CLR, family="Manrope, sans-serif", size=11),
        title_font=dict(color=_TITLE_CLR, size=title_size, family="Space Grotesk, sans-serif"),
        xaxis=dict(gridcolor=_GRID_CLR, zerolinecolor=_GRID_CLR, tickfont=dict(color=_FONT_CLR)),
        yaxis=dict(gridcolor=_GRID_CLR, zerolinecolor=_GRID_CLR, tickfont=dict(color=_FONT_CLR)),
        legend=dict(bgcolor="rgba(16,30,47,0.72)", bordercolor="rgba(88,192,218,0.24)", borderwidth=1, font=dict(color=_FONT_CLR)),
        margin=dict(l=10, r=10, t=45, b=10),
        hoverlabel=dict(bgcolor="#13263a", bordercolor="#58c0da", font_color="#e8f1f8"),
    )
    if height:
        upd["height"] = height
    upd.update(kwargs)
    fig.update_layout(**upd)
    return fig

if page == "📊 Overview":
    overview = get_overview()
    kpis = overview["kpis"]

    st.markdown('<div class="header-title">📊 Flipkart Sales Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    🎯 Welcome to the Flipkart Sales Analytics Dashboard! This interactive platform demonstrates advanced data science techniques 
    applied to e-commerce sales data. Explore customer segments, analyze sales trends, and make data-driven predictions.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Section with enhanced styling
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = kpis["total_revenue"]
        st.metric(
            label="💰 Total Revenue",
            value=f"₹{total_revenue:,.0f}",
            delta=f"₹{total_revenue/kpis['row_count']:,.0f} avg/order",
            delta_color="inverse"
        )
    
    with col2:
        total_orders = kpis["total_orders"]
        total_units = kpis["total_units"]
        st.metric(
            label="📦 Total Orders",
            value=f"{total_orders:,}",
            delta=f"{total_units:,} units sold",
            delta_color="inverse"
        )
    
    with col3:
        total_customers = kpis["total_customers"]
        avg_cust_revenue = total_revenue / total_customers
        st.metric(
            label="👥 Unique Customers",
            value=f"{total_customers:,}",
            delta=f"₹{avg_cust_revenue:,.0f} avg/customer",
            delta_color="inverse"
        )
    
    with col4:
        avg_discount = kpis["avg_discount"]
        avg_selling_price = kpis["avg_selling_price"]
        discount_percent = (avg_discount / avg_selling_price * 100) if avg_selling_price else 0.0
        st.metric(
            label="💹 Avg Discount",
            value=f"₹{avg_discount:,.0f}",
            delta=f"{discount_percent:.1f}% of price",
            delta_color="inverse"
        )
    
    # Interactive Charts Section
    st.markdown('<div class="section-header">📈 Sales Overview & Trends</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_revenue = overview["monthly_revenue"]
        fig = px.line(monthly_revenue, x="Month", y="Revenue",
                     title="Monthly Revenue Trend",
                     markers=True, line_shape="spline",
                     color_discrete_sequence=["#58c0da"])
        fig.update_traces(line=dict(width=2.5), marker=dict(size=7, color="#7ad4eb"))
        dark_layout(fig, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        monthly_quantity = overview["monthly_quantity"]
        fig = px.bar(monthly_quantity, x="Month", y="Quantity",
                    title="Monthly Quantity Sold",
                    color="Quantity",
                    color_continuous_scale="Tealgrn")
        dark_layout(fig, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution Analysis
    st.markdown('<div class="section-header">📊 Distribution Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(overview["hist_sample"], x="Revenue", nbins=50,
                          title="Revenue Distribution",
                          color_discrete_sequence=["#34d399"])
        dark_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(overview["hist_sample"], x="Quantity", nbins=30,
                          title="Quantity Distribution",
                          color_discrete_sequence=["#f87171"])
        dark_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(overview["hist_sample"], x="Discount", nbins=30,
                          title="Discount Distribution",
                          color_discrete_sequence=["#fbbf24"])
        dark_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Cities Comparison
    st.markdown('<div class="section-header">🏙️ Top Cities Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_cities_revenue = overview["top_cities_revenue"]
        fig = px.bar(x=top_cities_revenue["Revenue"], y=top_cities_revenue["city_name"],
                    title="Top 10 Cities by Revenue",
                    orientation='h',
                    color=top_cities_revenue["Revenue"],
                    color_continuous_scale="Tealgrn")
        dark_layout(fig, xaxis_title="Revenue (₹)", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_cities_orders = overview["top_cities_orders"]
        fig = px.bar(x=top_cities_orders["order_id"], y=top_cities_orders["city_name"],
                    title="Top 10 Cities by Orders",
                    orientation='h',
                    color=top_cities_orders["order_id"],
                    color_continuous_scale="Teal")
        dark_layout(fig, xaxis_title="Number of Orders", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Preview
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Showing **100** of **{kpis['row_count']:,}** records")
    with col2:
        show_all = st.checkbox("Show 200 Rows")
    
    if show_all:
        st.dataframe(overview["preview"], use_container_width=True)
    else:
        st.dataframe(overview["preview"].head(100), use_container_width=True)


elif page == "📈 EDA":
    df = get_df()
    st.markdown('<div class="header-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    🔍 Explore trends, patterns, and relationships in the Flipkart sales data through interactive visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Revenue Trends", "🏙️ City Analysis", "💰 Discount Impact", "🔗 Correlations", "📦 Product Analysis", "🎯 Advanced Filters"])
    
    with tab1:
        st.markdown('<div class="section-header">Monthly Revenue & Quantity Trends</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_data = df.groupby("Month").agg({"Revenue": "sum", "Quantity": "avg"}).reset_index()
            fig = px.line(monthly_data, x="Month", y="Revenue",
                         title="Revenue by Month",
                         markers=True, line_shape="spline",
                         color_discrete_sequence=["#58c0da"])
            fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
            dark_layout(fig, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(monthly_data, x="Month", y="Quantity",
                        title="Average Quantity by Month",
                        color="Quantity",
                        color_continuous_scale="Tealgrn")
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-Year Comparison
        st.markdown("**Year-over-Year Analysis**")
        yearly_data = df.groupby(["Year", "Month"])["Revenue"].sum().reset_index()
        fig = px.line(yearly_data, x="Month", y="Revenue", color="Year",
                     markers=True, line_shape="spline",
                     title="Revenue Comparison Across Years",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(line=dict(width=2))
        dark_layout(fig, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">City-wise Performance Analysis</div>', unsafe_allow_html=True)
        
        top_n = st.slider("Number of cities to display", min_value=5, max_value=20, value=10, key="city_slider")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_cities = df.groupby("city_name")["Revenue"].sum().sort_values(ascending=False).head(top_n)
            fig = px.bar(x=top_cities.values, y=top_cities.index,
                        title=f"Top {top_n} Cities by Revenue",
                        orientation='h',
                        color=top_cities.values,
                        color_continuous_scale="Tealgrn",
                        labels={"x": "Revenue (₹)", "y": "City"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            city_stats = df.groupby("city_name").agg({
                "Revenue": "sum",
                "order_id": "nunique",
                "Quantity": "sum",
                "Discount": "mean"
            }).sort_values("Revenue", ascending=False).head(top_n)
            city_stats.columns = ["Total Revenue", "Orders", "Units Sold", "Avg Discount"]
            st.dataframe(city_stats, use_container_width=True)
        
        # City Distribution Pie Chart
        st.markdown("**Revenue Distribution by Top Cities**")
        top_cities_pie = df.groupby("city_name")["Revenue"].sum().sort_values(ascending=False).head(10)
        fig = px.pie(values=top_cities_pie.values, names=top_cities_pie.index,
                    title="Revenue Share - Top 10 Cities",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">Discount Impact Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_sample = sample_data(df, n=3000)
            fig = px.scatter(df_sample, x="Discount", y="Revenue",
                           title="Discount vs Revenue Relationship",
                           color="Quantity",
                           size="Selling_Price",
                           color_continuous_scale="Plasma",
                           hover_data={"Discount": ":.2f", "Revenue": ":.2f"})
            dark_layout(fig, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            discount_bins = pd.cut(df["Discount"], bins=10)
            discount_impact = df.groupby(discount_bins, observed=True)["Revenue"].agg(["mean", "count"])
            fig = px.bar(x=range(len(discount_impact)), y=discount_impact["mean"],
                        title="Average Revenue by Discount Range",
                        color=discount_impact["mean"],
                        color_continuous_scale="Tealgrn",
                        labels={"x": "Discount Range", "y": "Average Revenue (₹)"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Impact Statistics
        st.markdown("**Discount Impact Metrics**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_discount = df["Discount"].mean()
            st.metric("Average Discount", f"₹{avg_discount:,.0f}")
        with col2:
            high_discount_revenue = df[df["Discount"] > df["Discount"].quantile(0.75)]["Revenue"].mean()
            st.metric("High Discount Avg Revenue", f"₹{high_discount_revenue:,.0f}")
        with col3:
            low_discount_revenue = df[df["Discount"] < df["Discount"].quantile(0.25)]["Revenue"].mean()
            st.metric("Low Discount Avg Revenue", f"₹{low_discount_revenue:,.0f}")
        with col4:
            correlation = df["Discount"].corr(df["Revenue"])
            st.metric("Discount-Revenue Correlation", f"{correlation:.3f}")
    
    with tab4:
        st.markdown('<div class="section-header">Feature Correlation Analysis</div>', unsafe_allow_html=True)
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d1321")
        ax.set_facecolor("#0d1321")
        corr_matrix = df[["Quantity", "Selling_Price", "Discount", "Revenue", "Landing_Price"]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".3f",
                   cbar_kws={"label": "Correlation", "shrink": 0.8}, ax=ax,
                   square=True, linewidths=1, linecolor="#1e293b",
                   vmin=-1, vmax=1,
                   annot_kws={"size": 11, "color": "white"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold",
                     pad=16, color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        plt.setp(ax.get_xticklabels(), color="#94a3b8")
        plt.setp(ax.get_yticklabels(), color="#94a3b8")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="#94a3b8")
        cbar.set_label("Correlation", color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("**Key Correlations:**")
        corr_with_revenue = corr_matrix["Revenue"].sort_values(ascending=False)
        for feature, corr in corr_with_revenue.items():
            if feature != "Revenue":
                direction = "📈 Positive" if corr > 0 else "📉 Negative"
                st.write(f"- **{feature}** → Revenue: {direction} ({corr:.3f})")
    
    with tab5:
        st.markdown('<div class="section-header">Product Sales Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_bins = pd.cut(df["Selling_Price"], bins=10)
            price_analysis = df.groupby(price_bins, observed=True).agg({"Revenue": "sum", "order_id": "nunique"})
            fig = px.bar(x=range(len(price_analysis)), y=price_analysis["Revenue"],
                        title="Revenue by Price Range",
                        color=price_analysis["Revenue"],
                        color_continuous_scale="Plasma",
                        labels={"y": "Total Revenue (₹)"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            quantity_bins = pd.cut(df["Quantity"], bins=10)
            quantity_analysis = df.groupby(quantity_bins, observed=True)["Revenue"].sum()
            fig = px.area(x=range(len(quantity_analysis)), y=quantity_analysis.values,
                         title="Revenue by Unit Quantity Ranges",
                         color_discrete_sequence=["#58c0da"],
                         labels={"y": "Total Revenue (₹)"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Product statistics
        st.markdown("**Product Price Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Price", f"₹{df['Selling_Price'].min():,.0f}")
        with col2:
            st.metric("Max Price", f"₹{df['Selling_Price'].max():,.0f}")
        with col3:
            st.metric("Avg Price", f"₹{df['Selling_Price'].mean():,.0f}")
        with col4:
            st.metric("Median Price", f"₹{df['Selling_Price'].median():,.0f}")
    
    with tab6:
        st.markdown('<div class="section-header">Advanced Data Filtering</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_year = st.selectbox("Select Year", sorted(df["Year"].unique()), key="year_select")
        with col2:
            selected_month = st.selectbox("Select Month", sorted(df["Month"].unique()), key="month_select")
        with col3:
            city_options = top_values(df["city_name"], n=250)
            selected_city = st.selectbox(
                "Select City (Top 250 by volume)",
                ["All"] + city_options,
                key="city_select",
            )
        
        # Apply filters
        filtered_df = df[(df["Year"] == selected_year) & (df["Month"] == selected_month)]
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df["city_name"] == selected_city]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(filtered_df))
        with col2:
            st.metric("Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")
        with col3:
            st.metric("Orders", filtered_df["order_id"].nunique())
        with col4:
            st.metric("Avg Order Value", f"₹{filtered_df['Revenue'].mean():,.0f}")
        
        st.write(f"**Filtered Data ({len(filtered_df)} records):**")
        st.dataframe(filtered_df.head(200), use_container_width=True, height=400)


elif page == "👥 Clustering":
    df = get_df()
    st.markdown('<div class="header-title">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    🎯 Understand customer behavior through advanced K-Means clustering. Identify customer segments based on purchasing patterns, 
    order frequency, and lifetime value to optimize marketing strategies.
    </div>
    """, unsafe_allow_html=True)
    
    cust_df = prepare_customer_df(df)
    plot_cust_df = sample_data(cust_df, n=12_000)
    
    # Clustering Configuration
    st.markdown('<div class="section-header">⚙️ Clustering Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    
    with col2:
        st.write(f"**Total Customers:** {len(cust_df):,}")
    
    with col3:
        st.write(f"**Avg Orders/Customer:** {cust_df['Total_Orders'].mean():.1f}")
    
    # Perform clustering
    scaler = StandardScaler()
    X = cust_df[["Quantity", "Revenue", "Discount", "Total_Orders"]]
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cust_df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # Cluster Statistics
    st.markdown('<div class="section-header">📊 Cluster Statistics</div>', unsafe_allow_html=True)
    
    cluster_stats = cust_df.groupby("Cluster").agg({
        "Quantity": ["sum", "mean"],
        "Revenue": ["sum", "mean"],
        "Total_Orders": ["sum", "mean"],
        "Discount": "mean"
    }).round(2)
    
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Visualizations
    st.markdown('<div class="section-header">📈 Cluster Visualizations</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📍 2D Scatter", "🔍 Detailed Analysis", "📊 Box Plots", "📋 Cluster Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(plot_cust_df, x="Revenue", y="Quantity",
                           color="Cluster",
                           size="Total_Orders",
                           hover_data=["Revenue", "Quantity", "Total_Orders", "Discount"],
                           title="Customer Segments (Revenue vs Quantity)",
                           color_discrete_sequence=px.colors.qualitative.Pastel,
                           labels={"Revenue": "Total Revenue (₹)", "Quantity": "Total Quantity"})
            dark_layout(fig, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(plot_cust_df, x="Total_Orders", y="Revenue",
                           color="Cluster",
                           size="Quantity",
                           hover_data=["Revenue", "Quantity", "Total_Orders", "Discount"],
                           title="Customer Segments (Orders vs Revenue)",
                           color_discrete_sequence=px.colors.qualitative.Pastel,
                           labels={"Total_Orders": "Total Orders", "Revenue": "Total Revenue (₹)"})
            dark_layout(fig, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Scatter
        st.markdown("**3D Cluster Visualization**")
        fig = px.scatter_3d(plot_cust_df, x="Revenue", y="Quantity", z="Total_Orders",
                           color="Cluster",
                           size="Discount",
                           title="3D Customer Segments",
                           color_discrete_sequence=px.colors.qualitative.Pastel,
                           labels={"Revenue": "Revenue (₹)", "Quantity": "Quantity", "Total_Orders": "Orders"})
        fig.update_traces(marker=dict(size=5))
        dark_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_size = cust_df["Cluster"].value_counts().sort_index()
            fig = px.bar(x=cluster_size.index, y=cluster_size.values,
                        title="Cluster Size Distribution",
                        color=cluster_size.values,
                        color_continuous_scale="Tealgrn",
                        labels={"x": "Cluster", "y": "Number of Customers"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cluster_revenue = cust_df.groupby("Cluster")["Revenue"].sum().sort_index()
            fig = px.bar(x=cluster_revenue.index, y=cluster_revenue.values,
                        title="Total Revenue by Cluster",
                        color=cluster_revenue.values,
                        color_continuous_scale="Teal",
                        labels={"x": "Cluster", "y": "Total Revenue (₹)"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Revenue Pie
        st.markdown("**Revenue Distribution by Cluster**")
        fig = px.pie(values=cluster_revenue.values, names=[f"Cluster {i}" for i in cluster_revenue.index],
                    title="Revenue Share by Cluster",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("**Box Plot Analysis by Cluster**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(plot_cust_df, x="Cluster", y="Revenue",
                        title="Revenue Distribution by Cluster",
                        color="Cluster",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(plot_cust_df, x="Cluster", y="Total_Orders",
                        title="Orders Distribution by Cluster",
                        color="Cluster",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Violin plot
        fig = px.violin(plot_cust_df, x="Cluster", y="Quantity",
                       title="Quantity Distribution by Cluster",
                       color="Cluster",
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       box=True, points="outliers")
        dark_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        selected_cluster = st.selectbox("Select Cluster to View", range(n_clusters))
        cluster_customers = cust_df[cust_df["Cluster"] == selected_cluster]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customers", len(cluster_customers))
        with col2:
            st.metric("Avg Revenue", f"₹{cluster_customers['Revenue'].mean():,.0f}")
        with col3:
            st.metric("Avg Orders", f"{cluster_customers['Total_Orders'].mean():.1f}")
        with col4:
            st.metric("Avg Discount", f"₹{cluster_customers['Discount'].mean():,.0f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"₹{cluster_customers['Revenue'].sum():,.0f}")
        with col2:
            st.metric("Total Orders", f"{cluster_customers['Total_Orders'].sum():,}")
        with col3:
            st.metric("Total Quantity", f"{cluster_customers['Quantity'].sum():,}")
        with col4:
            st.metric("% of Total", f"{(len(cluster_customers)/len(cust_df)*100):.1f}%")
        
        st.write(f"**Cluster {selected_cluster} Details ({len(cluster_customers)} customers):**")
        st.dataframe(cluster_customers.head(50), use_container_width=True)
    
    # Cluster Insights
    st.markdown('<div class="section-header">💡 Strategic Cluster Insights</div>', unsafe_allow_html=True)
    
    for cluster_id in range(n_clusters):
        cluster_data = cust_df[cust_df["Cluster"] == cluster_id]
        avg_revenue = cluster_data["Revenue"].mean()
        avg_orders = cluster_data["Total_Orders"].mean()
        cluster_size = len(cluster_data)
        avg_discount = cluster_data["Discount"].mean()
        
        if avg_revenue > cust_df["Revenue"].quantile(0.75):
            label = "🌟 HIGH-VALUE"
            color = "success"
        elif avg_revenue < cust_df["Revenue"].quantile(0.25):
            label = "⚠️ LOW-VALUE"
            color = "warning"
        else:
            label = "📊 MID-VALUE"
            color = "info"
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**Cluster {cluster_id}**")
        with col2:
            st.markdown(f"""
            **{label}** | Size: {cluster_size} customers ({(cluster_size/len(cust_df)*100):.1f}%) | 
            Avg Revenue: ₹{avg_revenue:,.0f} | Avg Orders: {avg_orders:.1f} | Avg Discount: ₹{avg_discount:,.0f}
            """)


elif page == "🔮 Prediction":
    df = get_df()
    model, r2, rmse = get_model_metrics()
    st.markdown('<div class="header-title">🔮 Revenue Prediction & Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    🎯 Use our trained Linear Regression model to predict revenue based on quantity, discount amount, and month. 
    Analyze profit trends and make informed business decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance Section
    st.markdown('<div class="section-header">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}", "✅ Higher is better", delta_color="off")
    with col2:
        st.metric("RMSE", f"₹{rmse:,.2f}", "⚠️ Lower is better", delta_color="off")
    with col3:
        st.metric("Mean Revenue", f"₹{df['Revenue'].mean():,.0f}", "Average per transaction", delta_color="off")
    
    st.markdown("""
    <div class="success-box">
    ✅ **Model Interpretation:**
    - **R² Score** indicates how well the model explains revenue variations (scale: 0-1)
    - **RMSE** shows the average prediction error in rupees
    - These metrics are calculated on hold-out test data for unbiased evaluation
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Section
    st.markdown('<div class="section-header">🎯 Interactive Revenue Prediction</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input(
            "Product Quantity", 
            min_value=0, 
            value=1,
            help="Number of units to purchase"
        )
    
    with col2:
        discount = st.number_input(
            "Discount Amount (₹)", 
            min_value=0.0, 
            value=0.0, 
            step=0.01,
            help="Discount in rupees"
        )
    
    with col3:
        month = st.selectbox(
            "Month", 
            list(range(1, 13)),
            help="Select month for prediction"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Predict Revenue", use_container_width=True, key="predict_btn"):
            features = np.array([[quantity, discount, month]])
            pred = model.predict(features)[0]
            
            confidence = min(max((r2 * 100), 0), 100)
            
            st.markdown(f"""
            <div class="success-box">
            <h3>🎯 Prediction Result</h3>
            <p><strong>Predicted Revenue: ₹{pred:,.2f}</strong></p>
            <p>Based on: {quantity} units, ₹{discount:,.2f} discount, Month {month}</p>
            <p>Model Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Input Summary:**")
        st.json({
            "Quantity": int(quantity),
            "Discount": float(f"{discount:.2f}"),
            "Month": int(month)
        })
    
    # What-If Analysis
    st.markdown('<div class="section-header">📊 What-If Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quantity Impact on Revenue**")
        quantities = np.arange(1, 21)
        base_discount = df["Discount"].mean()
        base_month = 6
        predictions_qty = [model.predict([[q, base_discount, base_month]])[0] for q in quantities]
        fig = px.line(x=quantities, y=predictions_qty,
                     title="Revenue vs Quantity",
                     markers=True,
                     color_discrete_sequence=["#58c0da"],
                     labels={"x": "Quantity", "y": "Predicted Revenue (₹)"})
        fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
        dark_layout(fig, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Discount Impact on Revenue**")
        discounts = np.arange(0, df["Discount"].max(), df["Discount"].max()/20)
        base_qty = 5
        predictions_dis = [model.predict([[base_qty, d, base_month]])[0] for d in discounts]
        fig = px.line(x=discounts, y=predictions_dis,
                     title="Revenue vs Discount",
                     markers=True,
                     color_discrete_sequence=["#34d399"],
                     labels={"x": "Discount (₹)", "y": "Predicted Revenue (₹)"})
        fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
        dark_layout(fig, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Impact
    st.markdown("**Monthly Seasonality Impact**")
    months = np.arange(1, 13)
    base_qty = 5
    base_dis = df["Discount"].mean()
    predictions_month = [model.predict([[base_qty, base_dis, m]])[0] for m in months]
    
    fig = px.bar(x=months, y=predictions_month,
                title="Seasonal Revenue Variation by Month",
                color=predictions_month,
                color_continuous_scale="Tealgrn",
                labels={"x": "Month", "y": "Predicted Revenue (₹)"})
    dark_layout(fig, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Profit Analysis Section
    st.markdown('<div class="section-header">💹 Comprehensive Profit Analysis</div>', unsafe_allow_html=True)
    
    df["Profit"] = df["Revenue"] - (df["Landing_Price"] * df["Quantity"])
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Monthly Profit", "🎯 Key Metrics", "🏙️ City Analysis", "💡 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_profit = df.groupby("Month")["Profit"].sum()
            fig = px.area(x=monthly_profit.index, y=monthly_profit.values,
                         title="Monthly Profit Trend",
                         color_discrete_sequence=["#34d399"],
                         labels={"x": "Month", "y": "Profit (₹)"})
            dark_layout(fig, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_revenue_comp = df.groupby("Month").agg({"Revenue": "sum", "Profit": "sum"})
            fig = px.bar(x=monthly_revenue_comp.index,
                        y=[monthly_revenue_comp["Revenue"], monthly_revenue_comp["Profit"]],
                        title="Revenue vs Profit by Month",
                        barmode='group',
                        color_discrete_map={"Revenue": "#58c0da", "Profit": "#34d399"},
                        labels={"x": "Month", "y": "Amount (₹)", "value": "Amount (₹)", "variable": "Type"})
            dark_layout(fig, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Profit", f"₹{df['Profit'].sum():,.0f}", delta=f"₹{df['Profit'].sum()/len(df):,.0f} per order")
        with col2:
            st.metric("Avg Profit/Order", f"₹{df['Profit'].mean():,.0f}")
        with col3:
            best_month_profit = df.groupby('Month')['Profit'].sum().max()
            st.metric("Best Month Profit", f"₹{best_month_profit:,.0f}")
        with col4:
            profit_margin = (df['Profit'].sum()/df['Revenue'].sum()*100)
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")
        with col2:
            st.metric("Total Cost", f"₹{(df['Landing_Price'] * df['Quantity']).sum():,.0f}")
        with col3:
            max_profit_order = df["Profit"].max()
            st.metric("Max Profit/Order", f"₹{max_profit_order:,.0f}")
        with col4:
            min_profit_order = df["Profit"].min()
            st.metric("Min Profit/Order", f"₹{min_profit_order:,.0f}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            city_profit = df.groupby("city_name")["Profit"].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=city_profit.values, y=city_profit.index,
                        title="Top 10 Cities by Profit",
                        orientation='h',
                        color=city_profit.values,
                        color_continuous_scale="Teal",
                        labels={"x": "Total Profit (₹)", "y": "City"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            city_profit_margin = (df.groupby("city_name")["Profit"].sum() /
                                 df.groupby("city_name")["Revenue"].sum() * 100).sort_values(ascending=False).head(10)
            fig = px.bar(x=city_profit_margin.values, y=city_profit_margin.index,
                        title="Profit Margin by City",
                        orientation='h',
                        color=city_profit_margin.values,
                        color_continuous_scale="Earth",
                        labels={"x": "Profit Margin (%)", "y": "City"})
            dark_layout(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        best_month = df.groupby("Month")["Profit"].sum().idxmax()
        worst_month = df.groupby("Month")["Profit"].sum().idxmin()
        best_city = df.groupby("city_name")["Profit"].sum().idxmax()
        worst_city = df.groupby("city_name")["Profit"].sum().idxmin()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            ✅ **Best Performing Segments**
            """,  unsafe_allow_html=True)
            
            st.write(f"• **Best Month:** Month {best_month} - ₹{df.groupby('Month')['Profit'].sum().max():,.0f} profit")
            st.write(f"• **Most Profitable City:** {best_city} - ₹{df.groupby('city_name')['Profit'].sum().max():,.0f} profit")
            st.write(f"• **Highest Profit Order:** ₹{df['Profit'].max():,.0f}")
            st.write(f"• **Best Profit Margin:** {(df.groupby('city_name')['Profit'].sum() / df.groupby('city_name')['Revenue'].sum() * 100).max():.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            ⚠️ **Areas for Improvement**
            """, unsafe_allow_html=True)
            
            st.write(f"• **Worst Month:** Month {worst_month} - ₹{df.groupby('Month')['Profit'].sum().min():,.0f} profit")
            st.write(f"• **Lowest Profit City:** {worst_city} - ₹{df.groupby('city_name')['Profit'].sum().min():,.0f} profit")
            st.write(f"• **Lowest Profit Order:** ₹{df['Profit'].min():,.0f}")
            st.write(f"• **Lowest Profit Margin:** {(df.groupby('city_name')['Profit'].sum() / df.groupby('city_name')['Revenue'].sum() * 100).min():.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("**💡 Strategic Recommendations:**")
        st.markdown("""
        1. **Focus on High-Margin Products:** Increase inventory for products in high-profit cities
        2. **Optimize Discounting:** Review discount strategy in low-profit months
        3. **Geographic Expansion:** Replicate successful strategies from best-performing cities
        4. **Seasonal Planning:** Prepare stock in advance for profitable months
        5. **Cost Optimization:** Analyze cost structure in underperforming cities
        """)
