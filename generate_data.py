"""
Script to generate a rich, realistic Loan Approval dataset.
Includes financial profiles, underwriting decisions, credit tiers, and realistic noise for data cleaning demonstration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

TOTAL_ROWS = 12500
NOISE_ROWS = 142

start_date = datetime(2025, 1, 1)
date_range_days = 365

regions = ["North Region", "South Region", "East Region", "West Region", "Central Hub"]
region_weights = [0.26, 0.24, 0.18, 0.22, 0.10]

employment_types = [
    "Salaried Professional",
    "Self-Employed",
    "Corporate Executive",
    "Public Sector",
    "Healthcare / Medical",
    "Freelancer / Consultant"
]
emp_weights = [0.38, 0.22, 0.12, 0.14, 0.08, 0.06]

home_ownership = ["MORTGAGE", "RENT", "OWN"]
home_weights = [0.48, 0.38, 0.14]

purposes = [
    "Debt Consolidation",
    "Home Improvement",
    "Business Expansion",
    "Auto Purchase",
    "Education",
    "Personal Needs",
    "Medical Emergency"
]
purpose_weights = [0.32, 0.22, 0.16, 0.12, 0.08, 0.06, 0.04]

terms = [12, 24, 36, 48, 60, 84]
term_weights = [0.08, 0.14, 0.35, 0.23, 0.15, 0.05]

data = []

for i in range(1, TOTAL_ROWS - NOISE_ROWS + 1):
    app_id = f"LN-2025-{i:05d}"
    
    # Application date
    days_offset = int(np.random.beta(2, 2) * date_range_days)
    app_date = start_date + timedelta(days=days_offset)
    
    # Age (21 - 68)
    age = int(np.clip(np.random.normal(38, 11), 21, 68))
    
    # Employment
    emp_type = np.random.choice(employment_types, p=emp_weights)
    emp_years = round(float(np.clip(np.random.exponential(5) + (age - 22) * 0.2, 0.5, 30)), 1)
    
    # Home ownership
    home = np.random.choice(home_ownership, p=home_weights)
    
    # Income (lognormal)
    base_income = np.random.lognormal(mean=11.1, sigma=0.52) # median around $66k
    income = round(float(np.clip(base_income, 22000, 320000)), -2)
    
    # Loan Purpose
    purpose = np.random.choice(purposes, p=purpose_weights)
    
    # Loan Amount
    if purpose == "Home Improvement":
        loan_amt = round(float(np.clip(np.random.normal(165000, 75000), 20000, 480000)), -2)
    elif purpose == "Business Expansion":
        loan_amt = round(float(np.clip(np.random.normal(210000, 95000), 30000, 500000)), -2)
    elif purpose == "Debt Consolidation":
        loan_amt = round(float(np.clip(np.random.normal(78000, 35000), 10000, 240000)), -2)
    elif purpose == "Auto Purchase":
        loan_amt = round(float(np.clip(np.random.normal(38000, 16000), 8000, 95000)), -2)
    else:
        loan_amt = round(float(np.clip(np.random.normal(45000, 25000), 5000, 150000)), -2)
        
    term_months = int(np.random.choice(terms, p=term_weights))
    
    # Credit Score (300 to 850)
    score = int(np.clip(np.random.normal(685, 82), 320, 850))
    
    # DTI Ratio (%)
    dti = round(float(np.clip(np.random.normal(32.5, 11.2), 6.5, 64.0)), 1)
    
    # Past defaults
    if score < 580:
        past_defaults = np.random.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
    elif score < 670:
        past_defaults = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
    else:
        past_defaults = np.random.choice([0, 1, 2], p=[0.96, 0.035, 0.005])
        
    # Collateral
    if home in ["OWN", "MORTGAGE"] and purpose in ["Home Improvement", "Business Expansion"]:
        collateral = round(float(loan_amt * np.random.uniform(0.7, 1.8)), -2)
    else:
        collateral = round(float(loan_amt * np.random.choice([0.0, 0.2, 0.5, 1.0], p=[0.55, 0.20, 0.15, 0.10])), -2)
        
    region = np.random.choice(regions, p=region_weights)
    
    # Credit Tier
    if score >= 800:
        credit_tier = "Exceptional (800-850)"
        base_rate = 5.8
        grade = "Grade A"
    elif score >= 740:
        credit_tier = "Very Good (740-799)"
        base_rate = 7.2
        grade = "Grade A"
    elif score >= 670:
        credit_tier = "Good (670-739)"
        base_rate = 9.4
        grade = "Grade B"
    elif score >= 580:
        credit_tier = "Fair (580-669)"
        base_rate = 14.1
        grade = "Grade C"
    else:
        credit_tier = "Poor (300-579)"
        base_rate = 19.8
        grade = "Grade D"
        
    interest_rate = round(base_rate + (dti / 100.0) * 4.5 + past_defaults * 2.8 + np.random.uniform(-0.5, 0.8), 2)
    interest_rate = float(np.clip(interest_rate, 5.2, 24.9))
    
    # Underwriting Decision Logic
    # Logistic credit score weight, DTI threshold penalty, default penalty
    z = (
        (score - 640) / 45.0
        - (dti - 35.0) / 8.5
        + (income / (loan_amt / (term_months / 12.0) + 1000) - 4.0) * 0.5
        + (0.8 if collateral >= loan_amt else 0.0)
        - past_defaults * 2.2
        + (0.4 if home == "OWN" else 0.0)
    )
    prob_approval = 1.0 / (1.0 + np.exp(-z))
    
    # Strict veto rules
    if score < 500 or dti > 54.0 or past_defaults >= 2:
        approved = False
    elif score >= 760 and dti < 38.0 and past_defaults == 0:
        approved = True
    else:
        approved = bool(np.random.binomial(1, np.clip(prob_approval, 0.02, 0.98)))
        
    # Rejection Reason
    if approved:
        rejection_reason = "N/A - Approved"
        loan_status = "Approved"
    else:
        loan_status = "Rejected"
        if past_defaults > 0 and (score < 620 or np.random.rand() > 0.4):
            rejection_reason = "Prior Credit Defaults"
        elif score < 600:
            rejection_reason = "Low Credit Score (< 600)"
        elif dti > 42.0:
            rejection_reason = "Excessive Debt-to-Income (DTI > 42%)"
        elif loan_amt > income * 3.8:
            rejection_reason = "Income Insufficient for Requested Loan Size"
        elif collateral < loan_amt * 0.3 and purpose in ["Business Expansion", "Home Improvement"]:
            rejection_reason = "Inadequate Collateral / Asset Backing"
        else:
            rejection_reason = "High Composite Risk Rating"
            
    # Risk Score (0-100, where 0 is safest, 100 is riskiest)
    risk_score = round(float(np.clip(100 - (prob_approval * 85 + (score - 300) / 550.0 * 15), 5, 95)), 1)
    
    if risk_score < 30:
        risk_segment = "Prime (Low Risk)"
    elif risk_score < 60:
        risk_segment = "Near-Prime (Moderate Risk)"
    else:
        risk_segment = "Subprime (High Risk)"
        
    data.append({
        "Application_ID": app_id,
        "Application_Date": app_date.strftime("%Y-%m-%d"),
        "Applicant_Age": age,
        "Annual_Income": income,
        "Employment_Type": emp_type,
        "Employment_Years": emp_years,
        "Home_Ownership": home,
        "Loan_Purpose": purpose,
        "Loan_Amount": loan_amt,
        "Loan_Term_Months": term_months,
        "Credit_Score": score,
        "Credit_Tier": credit_tier,
        "DTI_Ratio": dti,
        "Collateral_Value": collateral,
        "Past_Defaults": past_defaults,
        "Interest_Rate": interest_rate,
        "Branch_Region": region,
        "Loan_Grade": grade,
        "Loan_Status": loan_status,
        "Rejection_Reason": rejection_reason,
        "Risk_Score": risk_score,
        "Risk_Segment": risk_segment
    })

# Add noise records (duplicates, missing/negative values) to simulate real-world data cleaning
for k in range(NOISE_ROWS):
    dirty_rec = dict(data[k % 100])
    if k % 4 == 0:
        dirty_rec["Application_ID"] = dirty_rec["Application_ID"] # duplicate key
    elif k % 4 == 1:
        dirty_rec["Application_ID"] = f"LN-ERR-{k:04d}"
        dirty_rec["Annual_Income"] = -9999.0 # negative invalid income
    elif k % 4 == 2:
        dirty_rec["Application_ID"] = f"LN-ERR-{k:04d}"
        dirty_rec["Credit_Score"] = 9999 # impossible score
    else:
        dirty_rec["Application_ID"] = f"LN-ERR-{k:04d}"
        dirty_rec["Loan_Amount"] = None # missing value
    data.append(dirty_rec)

random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv("loan_approval_dataset.csv", index=False)
print(f"Generated {len(df)} records in loan_approval_dataset.csv")

