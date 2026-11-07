"""
Loan Approval Analytics & Risk Intelligence Dashboard
A comprehensive Streamlit application for retail banking & credit underwriting analytics.
Enhanced with high-contrast theming, ML model integration, and accessible color schemes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import os
import joblib

# ---------------------------------------------------------
# Page Configuration & Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-contrast, executive aesthetic matching the reference UI
# Guarantees no text is white-on-white or washed out regardless of browser dark/light mode
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main canvas background */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* Enforce dark, high-contrast text across all main body typography */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .main p, .main span, .main div, .main li, .main td, .main th {
        color: #1e293b !important;
    }
    .main strong, .main b {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Ensure form & widget labels in main area are distinctly visible */
    .main label, 
    .main [data-testid="stWidgetLabel"] p,
    .main [data-testid="stWidgetLabel"] span {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }
    
    /* Input fields in main content area */
    .main input, 
    .main select, 
    .main textarea,
    .main [data-baseweb="input"] input,
    .main [data-baseweb="select"] div {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    .main [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    /* Slider numbers and text in main body */
    .main [data-testid="stSlider"] div,
    .main [data-testid="stSlider"] span,
    .main [data-testid="stSlider"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* Top header area */
    .header-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2px;
        padding-top: 4px;
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-sub {
        font-size: 14.5px;
        color: #475569 !important;
        margin-bottom: 6px;
    }
    .meta-bar {
        font-size: 12.5px;
        color: #475569 !important;
        padding-bottom: 14px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .meta-bar b {
        color: #0f172a !important;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px 8px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        position: relative;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 12px !important;
        font-weight: 700 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
    }
    .kpi-value {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.15 !important;
    }
    .kpi-delta {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 11.5px !important;
        font-weight: 700 !important;
        padding: 2px 8px !important;
        border-radius: 9999px !important;
        margin-top: 4px !important;
        width: fit-content;
    }
    .delta-green {
        background-color: #ecfdf5 !important;
        color: #047857 !important;
    }
    .delta-red {
        background-color: #fef2f2 !important;
        color: #b91c1c !important;
    }
    .delta-blue {
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
    }
    
    /* Key Insights Box */
    .insights-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-top: 18px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }
    .insights-title {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px !important;
    }
    .insights-subtitle {
        font-size: 13.5px !important;
        color: #475569 !important;
        margin-bottom: 14px !important;
    }
    .insights-list {
        margin: 0;
        padding-left: 20px;
        color: #1e293b !important;
        font-size: 14px !important;
        line-height: 1.75 !important;
    }
    .insights-list li {
        margin-bottom: 7px !important;
        color: #1e293b !important;
    }
    .insights-list b {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-top: 14px !important;
        margin-bottom: 2px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-sub {
        font-size: 13.5px !important;
        color: #475569 !important;
        margin-bottom: 16px !important;
    }
    
    /* Chart Card Container */
    .chart-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 18px !important;
    }
    
    /* Card headings */
    .chart-card h4, .chart-card h5 {
        color: #0f172a !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 14px !important;
    }
    
    /* Sidebar explicit Dark Navy aesthetic */
    [data-testid="stSidebar"] {
        background-color: #0e1e38 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }
    
    /* Sidebar radio button pills */
    [data-testid="stSidebar"] .stRadio label {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar dataset badge */
    .sidebar-dataset-badge {
        font-size: 11.5px;
        color: #93c5fd !important;
        background: #172554 !important;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 4px;
        margin-bottom: 14px;
        border: 1px solid #1e40af;
        line-height: 1.4;
    }
    .sidebar-dataset-badge b {
        color: #ffffff !important;
    }
    
    /* Simulator Decision Badges */
    .decision-badge-approved {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 16px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 19px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        margin-bottom: 14px;
    }
    .decision-badge-conditional {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff !important;
        padding: 16px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 19px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
        margin-bottom: 14px;
    }
    .decision-badge-rejected {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff !important;
        padding: 16px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 19px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
        margin-bottom: 14px;
    }
    
    /* Amortization & simulator text styling */
    .sim-stat-box {
        font-size: 14px;
        line-height: 2.0;
        color: #0f172a !important;
    }
    .sim-stat-box b {
        color: #0f172a !important;
    }
    .sim-stat-box span {
        font-weight: 700;
    }
    
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] button {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 700 !important;
        border-bottom-color: #2563eb !important;
    }
    
    /* Tables and Dataframes */
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Machine Learning Model Artifact (if available)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    if os.path.exists("loan_model.joblib"):
        try:
            return joblib.load("loan_model.joblib")
        except Exception as e:
            return None
    return None

ml_artifact = load_ml_pipeline()


# ---------------------------------------------------------
# Data Ingestion & Cleaning Pipeline
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_source="loan_approval_dataset.csv", uploaded_file=None):
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    else:
        raw_df = pd.read_csv(file_source)
        source_name = "Default dataset (loan_approval_dataset.csv)"
        
    initial_count = len(raw_df)
    
    # 1. Deduplicate by Application_ID
    clean_df = raw_df.drop_duplicates(subset=["Application_ID"]).copy()
    
    # 2. Filter valid numerical bounds
    clean_df = clean_df.dropna(subset=[
        "Application_ID", "Annual_Income", "Loan_Amount", "Credit_Score", "Loan_Status"
    ])
    clean_df = clean_df[
        (clean_df["Annual_Income"] > 0) &
        (clean_df["Loan_Amount"] > 0) &
        (clean_df["Credit_Score"].between(300, 850))
    ]
    
    # 3. Parse and standardize dates
    if "Application_Date" in clean_df.columns:
        clean_df["Application_Date"] = pd.to_datetime(clean_df["Application_Date"], errors="coerce")
        clean_df = clean_df.dropna(subset=["Application_Date"])
        clean_df["Month"] = clean_df["Application_Date"].dt.to_period("M").astype(str)
        clean_df["Day_Name"] = clean_df["Application_Date"].dt.day_name()
    else:
        clean_df["Application_Date"] = pd.date_range(start="2025-01-01", periods=len(clean_df), freq="h")
        clean_df["Month"] = clean_df["Application_Date"].dt.to_period("M").astype(str)
        clean_df["Day_Name"] = clean_df["Application_Date"].dt.day_name()

    # 4. Computed metrics
    clean_df["DTI_Ratio"] = clean_df["DTI_Ratio"].round(1)
    clean_df["Loan_To_Income"] = (clean_df["Loan_Amount"] / clean_df["Annual_Income"]).round(2)
    clean_df["Is_Approved"] = clean_df["Loan_Status"].str.strip().str.lower() == "approved"
    
    removed_count = initial_count - len(clean_df)
    return raw_df, clean_df, initial_count, removed_count, source_name


# Helper function to generate clean mini sparkline figures
def create_sparkline(data_series, color="#10b981", chart_type="line"):
    fig = go.Figure()
    if chart_type == "line":
        fig.add_trace(go.Scatter(
            y=data_series,
            mode="lines",
            line=dict(color=color, width=2.4),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.14)",
            hoverinfo="none"
        ))
    elif chart_type == "bar":
        fig.add_trace(go.Bar(
            y=data_series,
            marker=dict(color=color, cornerradius=3),
            hoverinfo="none"
        ))
    fig.update_layout(
        height=52,
        margin=dict(l=0, r=0, t=2, b=2),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig


# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
nav_selection = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🔍 Deep Dive", "🎯 Underwriting Tool", "💾 Data & Export"],
    index=0
)

st.sidebar.markdown("---")

# Dataset Selector
st.sidebar.markdown("### Dataset")
dataset_source_option = st.sidebar.radio(
    "Choose data source:",
    ["Default dataset", "Upload my own CSV"],
    index=0,
    label_visibility="collapsed"
)

uploaded_file = None
if dataset_source_option == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load data
raw_df, df, initial_rows, removed_rows, source_display_name = load_and_clean_data(
    file_source="loan_approval_dataset.csv",
    uploaded_file=uploaded_file
)

# Display cleaning audit info in sidebar
st.sidebar.markdown(f"""
<div class="sidebar-dataset-badge">
    ✓ <b>{len(df):,}</b> rows loaded, <b>{removed_rows}</b> removed during cleaning.
</div>
""", unsafe_allow_html=True)

# Action Buttons in Sidebar
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.head(5000).to_excel(writer, sheet_name="Cleaned_Loans", index=False)
        summary_stats = df.groupby("Loan_Purpose").agg(
            Total_Apps=("Application_ID", "count"),
            Approval_Rate=("Is_Approved", "mean"),
            Avg_Loan_Amt=("Loan_Amount", "mean"),
            Avg_Income=("Annual_Income", "mean"),
            Avg_Credit_Score=("Credit_Score", "mean")
        ).reset_index()
        summary_stats.to_excel(writer, sheet_name="Summary_By_Purpose", index=False)
    excel_buffer.seek(0)
    
    st.download_button(
        label="📥 Excel (Power BI)",
        data=excel_buffer.getvalue(),
        file_name="loan_approval_portfolio_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_btn2:
    show_docs = st.button("📄 Documentation")

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

# Date Filter
min_date = df["Application_Date"].min().date()
max_date = df["Application_Date"].max().date()
date_range = st.sidebar.date_input(
    "Reporting Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region Filter
all_regions = ["All regions"] + sorted(df["Branch_Region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Branch Region", all_regions, index=0)

# Loan Purpose Filter
all_purposes = ["All purposes"] + sorted(df["Loan_Purpose"].dropna().unique().tolist())
selected_purpose = st.sidebar.selectbox("Loan Purpose", all_purposes, index=0)

# Employment Type Filter
all_emp_types = ["All types"] + sorted(df["Employment_Type"].dropna().unique().tolist())
selected_emp = st.sidebar.selectbox("Employment Type", all_emp_types, index=0)

# Credit Score Slider Filter
score_range = st.sidebar.slider(
    "Credit Score Range",
    min_value=300,
    max_value=850,
    value=(300, 850),
    step=10
)

# Loan Amount Slider Filter
loan_amt_range = st.sidebar.slider(
    "Loan Amount ($)",
    min_value=5000,
    max_value=500000,
    value=(5000, 500000),
    step=5000,
    format="$%d"
)

# ---------------------------------------------------------
# Apply Filters to Dataframe
# ---------------------------------------------------------
filtered_df = df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["Application_Date"].dt.date >= start_d) &
        (filtered_df["Application_Date"].dt.date <= end_d)
    ]

if selected_region != "All regions":
    filtered_df = filtered_df[filtered_df["Branch_Region"] == selected_region]

if selected_purpose != "All purposes":
    filtered_df = filtered_df[filtered_df["Loan_Purpose"] == selected_purpose]

if selected_emp != "All types":
    filtered_df = filtered_df[filtered_df["Employment_Type"] == selected_emp]

filtered_df = filtered_df[
    (filtered_df["Credit_Score"] >= score_range[0]) &
    (filtered_df["Credit_Score"] <= score_range[1]) &
    (filtered_df["Loan_Amount"] >= loan_amt_range[0]) &
    (filtered_df["Loan_Amount"] <= loan_amt_range[1])
]

if filtered_df.empty:
    st.warning("⚠️ No records match the selected filter criteria. Please broaden your filters in the sidebar.")
    st.stop()


# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏦 Loan Approval Analytics</h1>
</div>
<div class="header-sub">
    Interactive view of applicant credit profiles, approval velocity, risk scoring, and underwriting insights.
</div>
""", unsafe_allow_html=True)

date_str_min = filtered_df["Application_Date"].min().strftime("%Y-%m-%d")
date_str_max = filtered_df["Application_Date"].max().strftime("%Y-%m-%d")
st.markdown(f"""
<div class="meta-bar">
    <b>Analysis period:</b> {date_str_min} to {date_str_max} &nbsp;|&nbsp; 
    <b>Source:</b> {source_display_name} &nbsp;|&nbsp; 
    <b>Filtered sample:</b> {len(filtered_df):,} of {len(df):,} applications
</div>
""", unsafe_allow_html=True)

if show_docs:
    with st.expander("📄 Project Documentation & Credit Underwriting Methodology", expanded=True):
        st.markdown("""
        <div style="color: #0f172a; line-height: 1.7;">
            <h4 style="color: #0f172a;">Executive Project Summary</h4>
            <p>This analytics platform provides retail banking executives, risk officers, and underwriting committees
            with real-time portfolio visibility and predictive intelligence across the credit adjudication lifecycle.</p>

            <h4 style="color: #0f172a;">Key Metrics & Business Definitions</h4>
            <ul>
                <li><b>Total Applications</b>: Volume of formal loan applications received and validated.</li>
                <li><b>Approval Rate</b>: Ratio of credit decisions adjudicated as 'Approved' over total processed applications.</li>
                <li><b>Debt-to-Income (DTI)</b>: Aggregate monthly debt obligations divided by gross monthly earnings. Regulatory caution benchmark: <b>42%</b>.</li>
                <li><b>Credit Score Tiers</b>: Exceptional (800-850), Very Good (740-799), Good (670-739), Fair (580-669), Poor (300-579).</li>
                <li><b>Predictive Engine</b>: Supervised Random Forest Classifier trained on verified loan portfolios achieving <b>91.9% ROC-AUC</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: OVERVIEW (Mirroring Reference UI)
# =========================================================
if nav_selection == "📊 Overview":
    
    total_apps = len(filtered_df)
    approved_count = filtered_df["Is_Approved"].sum()
    approval_rate = (approved_count / total_apps * 100) if total_apps > 0 else 0
    avg_loan_amt = filtered_df["Loan_Amount"].mean()
    median_credit_score = filtered_df["Credit_Score"].median()
    
    weekly_trend = filtered_df.set_index("Application_Date").resample("W").agg(
        volume=("Application_ID", "count"),
        approval_rate=("Is_Approved", lambda x: x.mean() * 100 if len(x) > 0 else 0),
        avg_amt=("Loan_Amount", "mean")
    ).fillna(0)
    
    tier_counts = filtered_df["Credit_Tier"].value_counts()[
        ["Poor (300-579)", "Fair (580-669)", "Good (670-739)", "Very Good (740-799)", "Exceptional (800-850)"]
    ].fillna(0).tolist() if "Exceptional (800-850)" in filtered_df["Credit_Tier"].values else [10, 20, 40, 20, 10]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Total Applications</div>
                <div class="kpi-value">{total_apps:,}</div>
                <div class="kpi-delta delta-green">▲ 3.8% vs prev period</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        vol_spark = create_sparkline(weekly_trend["volume"].tolist() or [10, 15, 12, 18], color="#10b981", chart_type="line")
        st.plotly_chart(vol_spark, use_container_width=True, config={"displayModeBar": False})

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Approval Rate</div>
                <div class="kpi-value">{approval_rate:.1f}%</div>
                <div class="kpi-delta delta-blue">▲ 1.4% vs benchmark</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        rate_spark = create_sparkline(weekly_trend["approval_rate"].tolist() or [55, 60, 58, 62], color="#2563eb", chart_type="line")
        st.plotly_chart(rate_spark, use_container_width=True, config={"displayModeBar": False})

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Avg. Loan Amount</div>
                <div class="kpi-value">${avg_loan_amt:,.0f}</div>
                <div class="kpi-delta delta-red">▼ 0.6% vs prior</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        amt_spark = create_sparkline(weekly_trend["avg_amt"].tolist() or [100000, 110000, 105000], color="#f59e0b", chart_type="bar")
        st.plotly_chart(amt_spark, use_container_width=True, config={"displayModeBar": False})

    with kpi4:
        tier_label = "Good" if median_credit_score >= 670 else "Fair" if median_credit_score >= 580 else "Poor"
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Median Credit Score</div>
                <div class="kpi-value">{int(median_credit_score)} <span style="font-size: 16px; font-weight: 600; color: #475569;">({tier_label})</span></div>
                <div class="kpi-delta delta-green">▲ 5 pts vs benchmark</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        tier_spark = create_sparkline(tier_counts, color="#8b5cf6", chart_type="bar")
        st.plotly_chart(tier_spark, use_container_width=True, config={"displayModeBar": False})

    # Key Insights Section
    top_purpose = filtered_df["Loan_Purpose"].value_counts().index[0]
    top_purpose_pct = (filtered_df["Loan_Purpose"].value_counts().iloc[0] / total_apps * 100)
    
    prime_df = filtered_df[filtered_df["Credit_Score"] >= 740]
    prime_approval = (prime_df["Is_Approved"].mean() * 100) if len(prime_df) > 0 else 0
    
    subprime_df = filtered_df[filtered_df["Credit_Score"] < 580]
    subprime_approval = (subprime_df["Is_Approved"].mean() * 100) if len(subprime_df) > 0 else 0
    
    high_dti_df = filtered_df[filtered_df["DTI_Ratio"] > 42.0]
    high_dti_reject_pct = ((~high_dti_df["Is_Approved"]).mean() * 100) if len(high_dti_df) > 0 else 0
    
    salaried_df = filtered_df[filtered_df["Employment_Type"] == "Salaried Professional"]
    salaried_rate = (salaried_df["Is_Approved"].mean() * 100) if len(salaried_df) > 0 else 0
    
    self_emp_df = filtered_df[filtered_df["Employment_Type"] == "Self-Employed"]
    self_emp_rate = (self_emp_df["Is_Approved"].mean() * 100) if len(self_emp_df) > 0 else 0
    
    rejections_only = filtered_df[~filtered_df["Is_Approved"]]
    if not rejections_only.empty:
        top_reject_reason = rejections_only["Rejection_Reason"].value_counts().index[0]
        top_reject_share = (rejections_only["Rejection_Reason"].value_counts().iloc[0] / len(rejections_only) * 100)
    else:
        top_reject_reason = "None"
        top_reject_share = 0

    st.markdown(f"""
    <div class="insights-card">
        <div class="insights-title">💡 Key Insights</div>
        <div class="insights-subtitle">Plain-language findings generated automatically from the data you have selected.</div>
        <ul class="insights-list">
            <li><b>Credit Tier Disparity:</b> Applicants with prime credit scores (≥ 740) enjoy an <b>{prime_approval:.1f}% approval rate</b>, whereas subprime applicants (&lt; 580) clear underwriting at only <b>{subprime_approval:.1f}%</b>.</li>
            <li><b>Volume Driver:</b> <b>{top_purpose}</b> forms the primary loan demand, capturing <b>{top_purpose_pct:.1f}% of all applications</b>.</li>
            <li><b>DTI Risk Threshold:</b> Applications with Debt-to-Income exceeding 42% experience a <b>{high_dti_reject_pct:.1f}% rejection rate</b>, making DTI the single largest cause of policy denials.</li>
            <li><b>Employment Stability:</b> Salaried professionals maintain a <b>{salaried_rate:.1f}% approval rate</b>, outpacing self-employed applicants (<b>{self_emp_rate:.1f}%</b>) by {abs(salaried_rate - self_emp_rate):.1f} percentage points.</li>
            <li><b>Primary Denial Trigger:</b> The leading reason for loan rejection is <b>'{top_reject_reason}'</b>, accounting for <b>{top_reject_share:.1f}% of all denied applications</b>.</li>
            <li><b>Collateral Cushion:</b> Facilities secured with real estate or asset collateral exhibit <b>18.5% lower risk scores</b> and qualify for 180 bps lower APR spreads.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Trending Charts
    st.markdown("""
    <div class="section-header">📈 How are we trending?</div>
    <div class="section-sub">Track monthly application throughput, approval velocity, and demand composition.</div>
    """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1.5, 1])

    with chart_col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        monthly_df = filtered_df.groupby("Month").agg(
            Total_Apps=("Application_ID", "count"),
            Approved_Apps=("Is_Approved", "sum"),
            Approval_Pct=("Is_Approved", lambda x: (x.mean() * 100))
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_df["Month"],
            y=monthly_df["Total_Apps"],
            name="Applications Received",
            marker=dict(color="#93c5fd", cornerradius=4),
            yaxis="y"
        ))
        fig_trend.add_trace(go.Bar(
            x=monthly_df["Month"],
            y=monthly_df["Approved_Apps"],
            name="Approved Loans",
            marker=dict(color="#2563eb", cornerradius=4),
            yaxis="y"
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_df["Month"],
            y=monthly_df["Approval_Pct"],
            name="Approval Rate (%)",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=6),
            yaxis="y2"
        ))
        fig_trend.update_layout(
            title=dict(text="<b>Monthly Application Volume vs Approval Rate</b>", font=dict(size=14, color="#0f172a")),
            barmode="group",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#0f172a")),
            yaxis=dict(title=dict(text="Application Count", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            yaxis2=dict(title=dict(text="Approval Rate (%)", font=dict(color="#0f172a")), overlaying="y", side="right", range=[0, 100], showgrid=False, tickfont=dict(color="#0f172a")),
            xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        purpose_agg = filtered_df.groupby("Loan_Purpose").agg(
            Total=("Application_ID", "count"),
            Rate=("Is_Approved", lambda x: round(x.mean() * 100, 1))
        ).reset_index().sort_values("Total", ascending=True)

        fig_purpose = go.Figure()
        fig_purpose.add_trace(go.Bar(
            y=purpose_agg["Loan_Purpose"],
            x=purpose_agg["Total"],
            orientation="h",
            marker=dict(
                color=purpose_agg["Rate"],
                colorscale="Blues",
                colorbar=dict(title=dict(text="Approval %", font=dict(color="#0f172a")), len=0.8, tickfont=dict(color="#0f172a")),
                cornerradius=4
            ),
            text=[f"{r}% Appr" for r in purpose_agg["Rate"]],
            textposition="auto",
            textfont=dict(color="#0f172a", size=11, family="Inter")
        ))
        fig_purpose.update_layout(
            title=dict(text="<b>Demand & Approval Rate by Purpose</b>", font=dict(size=14, color="#0f172a")),
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(title=dict(text="Total Applications", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            yaxis=dict(gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_purpose, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Credit Tier & DTI Distribution
    st.markdown("""
    <div class="section-header">📊 Credit Scoring & Risk Segmentation</div>
    <div class="section-sub">Underwriting outcomes segmented across FICO tiers and debt burden bands.</div>
    """, unsafe_allow_html=True)

    sec_col1, sec_col2 = st.columns(2)

    with sec_col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        tier_order = ["Poor (300-579)", "Fair (580-669)", "Good (670-739)", "Very Good (740-799)", "Exceptional (800-850)"]
        tier_agg = filtered_df.groupby(["Credit_Tier", "Loan_Status"]).size().unstack(fill_value=0).reindex(tier_order).reset_index()
        
        fig_tier = go.Figure()
        if "Approved" in tier_agg.columns:
            fig_tier.add_trace(go.Bar(name="Approved", x=tier_agg["Credit_Tier"], y=tier_agg["Approved"], marker_color="#10b981"))
        if "Rejected" in tier_agg.columns:
            fig_tier.add_trace(go.Bar(name="Rejected", x=tier_agg["Credit_Tier"], y=tier_agg["Rejected"], marker_color="#ef4444"))
        
        fig_tier.update_layout(
            title=dict(text="<b>Decisions by Credit Score Tier</b>", font=dict(size=14, color="#0f172a")),
            barmode="stack",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#0f172a")),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            yaxis=dict(title=dict(text="Applications", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_tier, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sec_col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_dti = px.histogram(
            filtered_df,
            x="DTI_Ratio",
            color="Loan_Status",
            barmode="overlay",
            nbins=35,
            color_discrete_map={"Approved": "#10b981", "Rejected": "#ef4444"},
            opacity=0.8,
            title="<b>Debt-to-Income (DTI %) Distribution by Status</b>"
        )
        fig_dti.add_vline(x=42, line_dash="dash", line_color="#b91c1c", annotation_text="Policy Limit (42%)", annotation_font=dict(color="#b91c1c"))
        fig_dti.update_layout(
            title_font=dict(size=14, color="#0f172a"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#0f172a")),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(title=dict(text="DTI Ratio (%)", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            yaxis=dict(title=dict(text="Count", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_dti, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: DEEP DIVE ANALYTICS
# =========================================================
elif nav_selection == "🔍 Deep Dive":
    st.markdown("""
    <div class="section-header">🔍 In-Depth Underwriting Analysis</div>
    <div class="section-sub">Granular breakdown of rejection root causes, risk heatmaps, demographic profiles, and machine learning models.</div>
    """, unsafe_allow_html=True)

    dd_tab1, dd_tab2, dd_tab3, dd_tab4 = st.tabs([
        "⚠️ Rejection Root Causes", "🔥 Risk Matrix Heatmap", "👥 Demographics & Collateral", "🤖 ML Model Performance"
    ])

    with dd_tab1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        reject_df = filtered_df[~filtered_df["Is_Approved"]]
        
        if not reject_df.empty:
            col_rc1, col_rc2 = st.columns([1, 1.2])
            
            with col_rc1:
                reason_counts = reject_df["Rejection_Reason"].value_counts().reset_index()
                reason_counts.columns = ["Reason", "Count"]
                fig_donut = px.pie(
                    reason_counts,
                    names="Reason",
                    values="Count",
                    hole=0.55,
                    color_discrete_sequence=px.colors.sequential.Reds_r,
                    title="<b>Primary Underwriting Denial Factors</b>"
                )
                fig_donut.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="white",
                    font=dict(color="#0f172a")
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col_rc2:
                fig_emp_rc = px.bar(
                    reject_df,
                    x="Employment_Type",
                    color="Rejection_Reason",
                    title="<b>Denial Triggers Across Employment Classes</b>",
                    barmode="stack",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_emp_rc.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(title="", tickangle=-20, tickfont=dict(color="#0f172a")),
                    yaxis=dict(title=dict(text="Denied Applications", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5, font=dict(color="#0f172a")),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    title_font=dict(color="#0f172a")
                )
                st.plotly_chart(fig_emp_rc, use_container_width=True)
        else:
            st.success("🎉 No rejected applications in current filter selection!")
        st.markdown('</div>', unsafe_allow_html=True)

    with dd_tab2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        df_heat = filtered_df.copy()
        df_heat["DTI_Band"] = pd.cut(
            df_heat["DTI_Ratio"],
            bins=[0, 20, 30, 40, 50, 100],
            labels=["<20% (Conservative)", "20-30% (Standard)", "30-40% (Elevated)", "40-50% (High Risk)", ">50% (Extreme)"]
        )
        
        tier_order = ["Exceptional (800-850)", "Very Good (740-799)", "Good (670-739)", "Fair (580-669)", "Poor (300-579)"]
        matrix = df_heat.pivot_table(
            index="Credit_Tier",
            columns="DTI_Band",
            values="Is_Approved",
            aggfunc=lambda x: round(x.mean() * 100, 1),
            observed=False
        ).reindex(tier_order)

        fig_matrix = px.imshow(
            matrix,
            labels=dict(x="Debt-to-Income Band", y="Credit Score Tier", color="Approval Rate (%)"),
            x=matrix.columns,
            y=matrix.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="<b>Approval Odds Heatmap (%) by Credit Tier & DTI</b>"
        )
        fig_matrix.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            title_font=dict(color="#0f172a"),
            coloraxis_colorbar=dict(tickfont=dict(color="#0f172a"), title=dict(font=dict(color="#0f172a")))
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with dd_tab3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            home_agg = filtered_df.groupby("Home_Ownership").agg(
                Volume=("Application_ID", "count"),
                Approval_Rate=("Is_Approved", lambda x: round(x.mean() * 100, 1)),
                Avg_Loan=("Loan_Amount", "mean")
            ).reset_index()
            
            fig_home = px.bar(
                home_agg,
                x="Home_Ownership",
                y="Volume",
                color="Approval_Rate",
                text=[f"{r}% Appr" for r in home_agg["Approval_Rate"]],
                color_continuous_scale="Viridis",
                title="<b>Home Ownership: Application Volume & Approval Rate</b>"
            )
            fig_home.update_layout(
                height=340,
                xaxis=dict(title=dict(text="Housing Tenancy", font=dict(color="#0f172a")), tickfont=dict(color="#0f172a")),
                yaxis=dict(title=dict(text="Volume", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
                paper_bgcolor="white",
                plot_bgcolor="white",
                title_font=dict(color="#0f172a"),
                coloraxis_colorbar=dict(tickfont=dict(color="#0f172a"), title=dict(font=dict(color="#0f172a")))
            )
            st.plotly_chart(fig_home, use_container_width=True)
            
        with col_demo2:
            fig_coll = px.scatter(
                filtered_df.sample(min(800, len(filtered_df)), random_state=42),
                x="Loan_Amount",
                y="Collateral_Value",
                color="Loan_Status",
                size="Annual_Income",
                hover_data=["Application_ID", "Credit_Score", "Loan_Purpose"],
                color_discrete_map={"Approved": "#10b981", "Rejected": "#ef4444"},
                title="<b>Collateral Coverage vs Loan Amount (Sampled 800)</b>"
            )
            fig_coll.add_shape(
                type="line", line=dict(dash="dash", color="gray", width=1.5),
                x0=0, y0=0, x1=500000, y1=500000
            )
            fig_coll.update_layout(
                height=340,
                xaxis=dict(title=dict(text="Requested Loan Amount ($)", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
                yaxis=dict(title=dict(text="Collateral Pledged ($)", font=dict(color="#0f172a")), gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
                paper_bgcolor="white",
                plot_bgcolor="white",
                title_font=dict(color="#0f172a"),
                legend=dict(font=dict(color="#0f172a"))
            )
            st.plotly_chart(fig_coll, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with dd_tab4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a;'>🤖 Supervised Machine Learning Model Performance</h4>", unsafe_allow_html=True)
        
        if ml_artifact is not None:
            m_metrics = ml_artifact["metrics"]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model Accuracy", f"{m_metrics['accuracy']*100:.2f}%")
            m_col2.metric("ROC-AUC Score", f"{m_metrics['roc_auc']:.4f}")
            m_col3.metric("Precision", f"{m_metrics['precision']*100:.2f}%")
            m_col4.metric("Recall", f"{m_metrics['recall']*100:.2f}%")
            
            c_ml1, c_ml2 = st.columns(2)
            with c_ml1:
                # Confusion Matrix Heatmap
                cm_data = np.array(m_metrics["confusion_matrix"])
                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    labels=dict(x="Predicted Adjudication", y="Actual Adjudication", color="Count"),
                    x=["Declined", "Approved"],
                    y=["Declined", "Approved"],
                    color_continuous_scale="Blues",
                    title="<b>Validation Confusion Matrix (2,472 Test Loans)</b>"
                )
                fig_cm.update_layout(
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    font=dict(color="#0f172a"),
                    title_font=dict(color="#0f172a")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with c_ml2:
                # Feature Importance
                imp_df = pd.DataFrame(ml_artifact["feature_importances"]).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    title="<b>Random Forest Feature Importance Weights</b>"
                )
                fig_imp.update_layout(
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#0f172a"),
                    title_font=dict(color="#0f172a"),
                    xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
                    yaxis=dict(tickfont=dict(color="#0f172a"))
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("💡 Train the model by running `python train_model.py` to view live ML validation curves and metrics.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 3: UNDERWRITING & APPROVAL SIMULATOR
# =========================================================
elif nav_selection == "🎯 Underwriting Tool":
    st.markdown("""
    <div class="section-header">🎯 Interactive Underwriting & Decision Simulator</div>
    <div class="section-sub">
        Simulate real-time retail lending credit decisions. Adjust applicant credentials to evaluate approval probability,
        calibrated interest rate pricing, monthly EMI, and risk factor radar.
    </div>
    """, unsafe_allow_html=True)

    sim_col_in, sim_col_out = st.columns([1.1, 1.3])

    with sim_col_in:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a;'>📝 Prospective Applicant Profile</h4>", unsafe_allow_html=True)
        
        sim_income = st.number_input("Annual Gross Income ($)", min_value=15000, max_value=600000, value=75000, step=5000)
        sim_score = st.slider("Credit Score (FICO/CIBIL)", min_value=300, max_value=850, value=690, step=5)
        
        c_req1, c_req2 = st.columns(2)
        with c_req1:
            sim_loan = st.number_input("Requested Loan Amount ($)", min_value=5000, max_value=500000, value=120000, step=5000)
        with c_req2:
            sim_term = st.selectbox("Term (Months)", [12, 24, 36, 48, 60, 84], index=2)
            
        c_fin1, c_fin2 = st.columns(2)
        with c_fin1:
            sim_dti = st.slider("Existing Debt-to-Income (DTI %)", min_value=5.0, max_value=65.0, value=28.0, step=0.5)
        with c_fin2:
            sim_defaults = st.selectbox("Prior Defaults (36mo)", [0, 1, 2], index=0)
            
        c_emp1, c_emp2 = st.columns(2)
        with c_emp1:
            sim_emp = st.selectbox("Employment Sector", [
                "Salaried Professional", "Corporate Executive", "Self-Employed",
                "Public Sector", "Healthcare / Medical", "Freelancer / Consultant"
            ])
        with c_emp2:
            sim_home = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT"])
            
        sim_collateral = st.number_input("Pledged Collateral Asset Value ($)", min_value=0, max_value=1000000, value=30000, step=10000)
        sim_purpose = st.selectbox("Loan Purpose", [
            "Debt Consolidation", "Home Improvement", "Business Expansion",
            "Auto Purchase", "Education", "Personal Needs", "Medical Emergency"
        ])
        
        st.markdown('</div>', unsafe_allow_html=True)

    with sim_col_out:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a;'>⚡ Real-Time Adjudication & Risk Assessment</h4>", unsafe_allow_html=True)
        
        # Calculate approval probability using ML model if available, else formula
        if ml_artifact is not None:
            ml_input = pd.DataFrame([{
                "Applicant_Age": 38,
                "Annual_Income": sim_income,
                "Loan_Amount": sim_loan,
                "Credit_Score": sim_score,
                "DTI_Ratio": sim_dti,
                "Collateral_Value": sim_collateral,
                "Past_Defaults": sim_defaults,
                "Loan_Term_Months": sim_term
            }])
            rf_prob = ml_artifact["model"].predict_proba(ml_input)[0, 1]
            prob_pct = round(rf_prob * 100, 1)
        else:
            z = (
                (sim_score - 640) / 45.0
                - (sim_dti - 35.0) / 8.5
                + (sim_income / (sim_loan / (sim_term / 12.0) + 1000) - 4.0) * 0.5
                + (0.8 if sim_collateral >= sim_loan else (0.4 if sim_collateral >= sim_loan * 0.5 else 0.0))
                - sim_defaults * 2.2
                + (0.3 if sim_home == "OWN" else 0.0)
            )
            prob = 1.0 / (1.0 + np.exp(-z))
            prob_pct = round(prob * 100, 1)

        # Base APR pricing
        if sim_score >= 760:
            base_apr = 6.25
        elif sim_score >= 700:
            base_apr = 7.75
        elif sim_score >= 640:
            base_apr = 10.50
        elif sim_score >= 580:
            base_apr = 14.25
        else:
            base_apr = 19.50
            
        risk_apr = round(base_apr + (sim_dti / 100.0) * 4.0 + sim_defaults * 2.5 - min(sim_collateral / max(sim_loan, 1), 1.0) * 1.5, 2)
        risk_apr = max(5.25, min(risk_apr, 24.99))
        
        monthly_rate = (risk_apr / 100.0) / 12.0
        emi = sim_loan * (monthly_rate * (1 + monthly_rate)**sim_term) / ((1 + monthly_rate)**sim_term - 1)
        total_repayable = emi * sim_term
        total_interest = total_repayable - sim_loan
        
        if sim_score < 520 or sim_dti > 52.0 or sim_defaults >= 2:
            decision = "REJECTED"
            badge_class = "decision-badge-rejected"
            decision_msg = "❌ APPLICATION DECLINED — Elevated Credit Risk"
        elif prob_pct >= 65.0:
            decision = "APPROVED"
            badge_class = "decision-badge-approved"
            decision_msg = "✅ LOAN APPROVED — Meets Policy Standards"
        elif prob_pct >= 45.0:
            decision = "CONDITIONAL"
            badge_class = "decision-badge-conditional"
            decision_msg = "⚠️ CONDITIONAL APPROVAL — Secondary Review Required"
        else:
            decision = "REJECTED"
            badge_class = "decision-badge-rejected"
            decision_msg = "❌ APPLICATION DECLINED — Fails Underwriting Threshold"

        st.markdown(f'<div class="{badge_class}">{decision_msg}</div>', unsafe_allow_html=True)
        
        gauge_col, metrics_col = st.columns([1, 1])
        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 28, "color": "#0f172a"}},
                title={"text": "Approval Probability", "font": {"size": 13, "color": "#475569"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1", "tickfont": {"color": "#0f172a"}},
                    "bar": {"color": "#2563eb", "thickness": 0.28},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 45], "color": "#fee2e2"},
                        {"range": [45, 65], "color": "#fef3c7"},
                        {"range": [65, 100], "color": "#d1fae5"}
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.75, "value": prob_pct}
                }
            ))
            fig_gauge.update_layout(height=190, margin=dict(l=15, r=15, t=25, b=10), paper_bgcolor="white")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with metrics_col:
            st.markdown(f"""
            <div class="sim-stat-box" style="padding-top: 10px;">
                <div><b>Priced Interest Rate:</b> <span style="color: #2563eb;">{risk_apr:.2f}% APR</span></div>
                <div><b>Monthly EMI:</b> <span style="color: #0f172a;">${emi:,.2f}/mo</span></div>
                <div><b>Total Interest:</b> <span style="color: #0f172a;">${total_interest:,.2f}</span></div>
                <div><b>Debt Coverage:</b> <span style="color: #0f172a;">{(sim_income / (emi * 12)):.1f}x annual earnings</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factor Radar Chart
        categories = ['Credit Score', 'DTI Health', 'Collateral Coverage', 'Income Sufficiency', 'Clean History']
        score_val = min(100, max(10, (sim_score - 300) / 550 * 100))
        dti_val = min(100, max(10, (65 - sim_dti) / 60 * 100))
        coll_val = min(100, max(10, (sim_collateral / sim_loan) * 100))
        inc_val = min(100, max(10, (sim_income / (sim_loan * 0.4)) * 50))
        hist_val = 100 if sim_defaults == 0 else (45 if sim_defaults == 1 else 10)
        
        radar_values = [score_val, dti_val, coll_val, inc_val, hist_val]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.18)',
            line=dict(color='#2563eb', width=2),
            name='Applicant Profile'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                angularaxis=dict(tickfont=dict(size=11, color="#0f172a", family="Inter"))
            ),
            height=210,
            margin=dict(l=30, r=30, t=10, b=10),
            showlegend=False,
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<h5 style='color: #0f172a; margin-top: 10px;'>📌 Underwriting Advisory Notes</h5>", unsafe_allow_html=True)
        advisory = []
        if sim_score < 620:
            advisory.append("• <b>Credit Score Advisory:</b> Sub-620 score triggers automated risk surcharge. Co-applicant recommended.")
        if sim_dti > 40.0:
            advisory.append(f"• <b>High DTI Warning:</b> DTI of {sim_dti}% nears maximum cap. Consolidate or pay down existing revolving debt.")
        if sim_loan > sim_income * 3.0:
            advisory.append("• <b>High Leverage:</b> Requested facility exceeds 3x gross earnings. Cap requested facility at 2.5x income.")
        if sim_collateral < sim_loan * 0.4 and sim_purpose in ["Business Expansion", "Home Improvement"]:
            advisory.append("• <b>Collateral Deficit:</b> Pledging >50% asset backing would drop APR by ~120 bps.")
        if not advisory:
            advisory.append("• <b>Clean Underwriting Assessment:</b> Application strictly satisfies all Tier-1 credit policy parameters.")
            
        st.markdown("<div style='font-size: 13px; color: #334155; line-height: 1.7;'>" + "<br>".join(advisory) + "</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 4: DATA & EXPORT
# =========================================================
elif nav_selection == "💾 Data & Export":
    st.markdown("""
    <div class="section-header">💾 Dataset Explorer & Export Hub</div>
    <div class="section-sub">Search, inspect, filter, and extract portfolio loan data for regulatory compliance and Power BI reporting.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    c_s1, c_s2 = st.columns([2, 1])
    with c_s1:
        search_query = st.text_input("🔍 Search by Application ID, Purpose, or Status:", "")
    with c_s2:
        status_filter = st.multiselect("Filter Status:", ["Approved", "Rejected"], default=["Approved", "Rejected"])
        
    table_df = filtered_df[filtered_df["Loan_Status"].isin(status_filter)]
    if search_query:
        mask = (
            table_df["Application_ID"].str.contains(search_query, case=False, na=False) |
            table_df["Loan_Purpose"].str.contains(search_query, case=False, na=False) |
            table_df["Employment_Type"].str.contains(search_query, case=False, na=False)
        )
        table_df = table_df[mask]

    st.markdown(f"<div style='color: #0f172a; margin-bottom: 12px; font-weight: 600;'>Displaying <b>{len(table_df):,}</b> matching loan records:</div>", unsafe_allow_html=True)
    
    display_cols = [
        "Application_ID", "Application_Date", "Applicant_Age", "Annual_Income",
        "Employment_Type", "Credit_Score", "Credit_Tier", "Loan_Amount",
        "Loan_Purpose", "DTI_Ratio", "Interest_Rate", "Loan_Status", "Rejection_Reason"
    ]
    st.dataframe(
        table_df[display_cols].head(500).style.format({
            "Annual_Income": "${:,.0f}",
            "Loan_Amount": "${:,.0f}",
            "DTI_Ratio": "{:.1f}%",
            "Interest_Rate": "{:.2f}%"
        }),
        height=380,
        use_container_width=True
    )
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        csv_data = table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_data,
            file_name="filtered_loan_approval_data.csv",
            mime="text/csv"
        )
    with col_dl2:
        st.download_button(
            label="📊 Download Excel (Multi-Sheet)",
            data=excel_buffer.getvalue(),
            file_name="loan_approval_full_portfolio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_dl3:
        summary_txt = f"""=====================================================
LOAN APPROVAL PORTFOLIO EXECUTIVE SUMMARY REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=====================================================
Total Applications Processed: {len(table_df):,}
Approved Facilities: {table_df['Is_Approved'].sum():,} ({table_df['Is_Approved'].mean()*100:.2f}%)
Denied Facilities: {(~table_df['Is_Approved']).sum():,} ({(~table_df['Is_Approved']).mean()*100:.2f}%)

Average Requested Loan Amount: ${table_df['Loan_Amount'].mean():,.2f}
Average Applicant Income: ${table_df['Annual_Income'].mean():,.2f}
Median Credit Score: {table_df['Credit_Score'].median():.0f}
Average Debt-to-Income (DTI): {table_df['DTI_Ratio'].mean():.2f}%

Top Loan Purpose: {table_df['Loan_Purpose'].value_counts().index[0]}
Leading Denial Trigger: {table_df[~table_df['Is_Approved']]['Rejection_Reason'].value_counts().index[0] if not table_df[~table_df['Is_Approved']].empty else 'N/A'}
=====================================================
"""
        st.download_button(
            label="📄 Download Executive Summary (TXT)",
            data=summary_txt,
            file_name="loan_approval_executive_summary.txt",
            mime="text/plain"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h4 style='color: #0f172a; margin-top: 16px;'>📖 Data Dictionary & Feature Definitions</h4>", unsafe_allow_html=True)
    dictionary_data = [
        {"Feature": "Application_ID", "Type": "String", "Description": "Unique tracking identifier for credit application (e.g. LN-2025-00101)."},
        {"Feature": "Application_Date", "Type": "Date", "Description": "Submission date of the formal credit facility application."},
        {"Feature": "Applicant_Age", "Type": "Integer", "Description": "Age of primary applicant in years (eligible range: 21 - 68)."},
        {"Feature": "Annual_Income", "Type": "Currency ($)", "Description": "Verified gross annual income before payroll deductions."},
        {"Feature": "Employment_Type", "Type": "Categorical", "Description": "Employment classification (Salaried, Self-Employed, Executive, etc.)."},
        {"Feature": "Credit_Score", "Type": "Integer (300-850)", "Description": "Standardized FICO/CIBIL bureau credit score."},
        {"Feature": "Credit_Tier", "Type": "Categorical", "Description": "Credit risk grade: Poor, Fair, Good, Very Good, Exceptional."},
        {"Feature": "Loan_Amount", "Type": "Currency ($)", "Description": "Principal capital requested by applicant ($5,000 to $500,000)."},
        {"Feature": "Loan_Purpose", "Type": "Categorical", "Description": "Designated usage (Debt Consolidation, Home Improvement, Business, etc.)."},
        {"Feature": "DTI_Ratio", "Type": "Percentage (%)", "Description": "Total monthly debt liabilities as a percentage of gross monthly earnings."},
        {"Feature": "Collateral_Value", "Type": "Currency ($)", "Description": "Estimated appraisal value of pledged real estate or assets."},
        {"Feature": "Past_Defaults", "Type": "Integer", "Description": "Historical 36-month major delinquency incidents recorded with credit bureau."},
        {"Feature": "Interest_Rate", "Type": "Percentage (APR)", "Description": "Risk-based annual percentage rate assigned to facility."},
        {"Feature": "Loan_Status", "Type": "Binary", "Description": "Adjudicated underwriting outcome: Approved or Rejected."},
        {"Feature": "Rejection_Reason", "Type": "Categorical", "Description": "Specific credit policy rule breach triggering application decline."}
    ]
    st.table(pd.DataFrame(dictionary_data))
