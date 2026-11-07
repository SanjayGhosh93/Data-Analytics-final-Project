# 🏦 Loan Approval Analytics & Risk Intelligence

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/SanjayGhosh93/Data-Analytics-final-Project)
[![Streamlit App](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen?logo=python)](https://python.org)

An interactive, executive-grade retail banking and credit underwriting dashboard built with **Streamlit**, **Plotly**, and **Pandas**. 

This project mirrors the high-aesthetic executive analytics architecture (inspired by enterprise KPI monitoring and footfall analytics), reimagined specifically for **Retail Lending, Credit Risk Adjudication, and Loan Approval Analysis**.

---

## 🌟 Key Features

### 1. 📊 Executive Overview Dashboard
* **4 Executive KPI Summary Cards**:
  * **Total Applications**: Volume counter with period-over-period growth delta and dynamic weekly sparkline.
  * **Approval Rate (%)**: Real-time adjudication approval velocity curve.
  * **Avg. Loan Amount ($)**: Capital size tracking with distribution mini-bars.
  * **Median Credit Score**: FICO/CIBIL credit tier gauge and benchmark tracking.
* **💡 Plain-Language Key Insights**: Automated dynamic analytical findings derived on the fly from user filters (Credit tier disparities, DTI risk thresholds, employment stability impact, and top rejection triggers).
* **📈 Trending & Trajectory**: Monthly throughput vs approval velocity dual-axis chart with zoom controls.
* **📊 Risk & Purpose Segmentation**: Approval rate across loan purposes, stacked FICO tier adjudications, and DTI density distributions.

### 2. 🔍 Deep Dive Analytics
* **Underwriting Denial Root Cause Analysis**: Donut and class-stratified breakdown of why applications get declined.
* **Risk Heatmap Matrix**: 2D heat-matrix of approval odds across Credit Score Tiers vs Debt-to-Income (DTI) Bands.
* **Demographics & Collateral Backing**: Impact of housing tenancy (Own vs Rent vs Mortgage) and asset collateral coverage.

### 3. 🎯 Interactive Underwriting & Decision Simulator
* Simulates real-time credit underwriting decisions:
  * Dynamic inputs: Income, Credit Score, Requested Amount, Term, DTI %, Prior Defaults, Collateral Asset Value, and Sector.
  * Instant Underwriting Adjudication: **Approved**, **Conditional Review**, or **Declined**.
  * Dynamic Approval Probability Gauge needle.
  * Risk-based APR interest rate quotation and Monthly Payment (EMI) amortization calculator.
  * Multi-dimensional Risk Factor Radar chart comparing applicant profile against portfolio standards.
  * Actionable Underwriting Advisory notes for credit repair or structuring approval.

### 4. 💾 Data Explorer & Multi-Format Export
* Live searchable table with status filters and currency formatting.
* **One-Click Export**:
  * Filtered CSV download.
  * Formatted Multi-Sheet Excel workbook (`.xlsx`) designed for direct import into **Power BI**, **Tableau**, or Excel Pivot Tables.
  * Executive Summary Text Report (`.txt`).
* Full Data Dictionary with type descriptions and regulatory context.

---

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml             # Custom high-contrast executive theme configuration
├── app.py                      # Main Streamlit dashboard application
├── generate_data.py            # Dataset generation script (12,500 records with realistic financial distributions)
├── loan_approval_dataset.csv   # Primary default dataset
├── loan_approval_eda_and_modeling.ipynb  # End-to-end Jupyter Notebook with EDA & modeling
├── loan_model.joblib           # Pre-trained Random Forest ML model artifact (91.9% ROC-AUC)
├── train_model.py              # ML pipeline training and serialization script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion rules
└── README.md                   # Complete project documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate Default Dataset
```bash
python generate_data.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 📊 Dataset Schema Overview

| Feature | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| **Application_ID** | String | `LN-2025-XXXXX` | Unique identifier per application |
| **Application_Date** | Date | 2025-01-01 to 2025-12-31 | Date of application submission |
| **Applicant_Age** | Integer | 21 – 68 | Age of the primary applicant |
| **Annual_Income** | Currency | $22,000 – $320,000 | Verified annual earnings |
| **Employment_Type** | Categorical | Salaried, Self-Employed, etc. | Professional classification |
| **Credit_Score** | Integer | 300 – 850 | FICO/CIBIL credit bureau rating |
| **Credit_Tier** | Categorical | Poor to Exceptional | Risk grading classification |
| **Loan_Amount** | Currency | $5,000 – $500,000 | Requested principal facility size |
| **Loan_Purpose** | Categorical | Debt Consolidation, Home Imp, etc. | Capital deployment objective |
| **DTI_Ratio** | Percentage | 6.5% – 64.0% | Debt-to-Income liability ratio |
| **Collateral_Value** | Currency | $0 – $750,000 | Pledged asset appraisal value |
| **Loan_Status** | Binary | Approved / Rejected | Underwriting adjudication outcome |
| **Rejection_Reason** | Categorical | DTI, Score, Defaults, etc. | Policy basis for declined loans |

