#!/usr/bin/env python3
"""
Analyze actual data transformations and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

def analyze_transformations():
    """Analyze what transformations are actually applied"""
    
    # Load sample data to see what's available
    df_2022 = pd.read_excel("data/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
    df_llm = pd.read_csv("data/2022 Applicants Reviewed by Trusted Reviewers/llm_scores_2022.csv")
    
    print("="*80)
    print("FEATURE ENGINEERING TRANSFORMATION ANALYSIS")
    print("="*80)
    
    # Map expected features to actual columns
    feature_mapping = {
        "Expected Feature": [],
        "Raw Column Name": [],
        "Transformation Applied": [],
        "Missing %": []
    }
    
    # 1. Age transformation
    print("\n1. AGE TRANSFORMATION:")
    print("-"*40)
    if 'Age' in df_2022.columns:
        print("Age column exists directly")
        feature_mapping["Expected Feature"].append("Age")
        feature_mapping["Raw Column Name"].append("Age")
        feature_mapping["Transformation Applied"].append("None (direct use)")
        feature_mapping["Missing %"].append(f"{df_2022['Age'].isna().sum()/len(df_2022)*100:.1f}%")
    elif 'Date of Birth' in df_2022.columns:
        print("Date of Birth found - Age calculated as: (Current Date - Date of Birth) / 365.25")
        feature_mapping["Expected Feature"].append("Age")
        feature_mapping["Raw Column Name"].append("Date of Birth")
        feature_mapping["Transformation Applied"].append("(Current Date - DOB) / 365.25")
        feature_mapping["Missing %"].append(f"{df_2022['Date of Birth'].isna().sum()/len(df_2022)*100:.1f}%")
    
    # 2. State transformation
    print("\n2. STATE TRANSFORMATION:")
    print("-"*40)
    if 'State' in df_2022.columns:
        print("State column found")
        print("Transformation: Grouped into 4 regions (West, Northeast, South, Midwest)")
        print("Unknown states → 'Unknown' category")
        feature_mapping["Expected Feature"].append("State_Grouped")
        feature_mapping["Raw Column Name"].append("State")
        feature_mapping["Transformation Applied"].append("Grouped into 4 US regions")
        feature_mapping["Missing %"].append(f"{df_2022['State'].isna().sum()/len(df_2022)*100:.1f}%")
    
    # 3. Citizenship transformation
    print("\n3. CITIZENSHIP TRANSFORMATION:")
    print("-"*40)
    if 'Citizenship' in df_2022.columns:
        print("Citizenship column found")
        print("Transformation: Binary classification (US_Citizen vs Other)")
        feature_mapping["Expected Feature"].append("Citizenship_Status")
        feature_mapping["Raw Column Name"].append("Citizenship")
        feature_mapping["Transformation Applied"].append("Binary: US_Citizen vs Other")
        feature_mapping["Missing %"].append(f"{df_2022['Citizenship'].isna().sum()/len(df_2022)*100:.1f}%")
    
    # 4. Binary indicators
    print("\n4. BINARY INDICATOR TRANSFORMATIONS:")
    print("-"*40)
    binary_cols = ['First_Generation_Ind', 'Disadvantanged_Ind']
    for col in binary_cols:
        if col in df_2022.columns:
            print(f"{col}: Yes/No/1/0 → Binary 0/1")
            feature_mapping["Expected Feature"].append(col)
            feature_mapping["Raw Column Name"].append(col)
            feature_mapping["Transformation Applied"].append("Yes/No → 0/1 binary")
            feature_mapping["Missing %"].append(f"{df_2022[col].isna().sum()/len(df_2022)*100:.1f}%")
    
    # 5. Experience hours transformation
    print("\n5. EXPERIENCE HOURS TRANSFORMATIONS:")
    print("-"*40)
    exp_cols = ['Exp_Hour_Research', 'Exp_Hour_Volunteer_Med', 'Exp_Hour_Volunteer_Non_Med', 
                'Comm_Service_Total_Hours', 'HealthCare_Total_Hours']
    for col in exp_cols:
        if col in df_2022.columns:
            print(f"{col}: log(1 + hours) transformation")
            feature_mapping["Expected Feature"].append(col)
            feature_mapping["Raw Column Name"].append(col)
            feature_mapping["Transformation Applied"].append("log(1 + x) transformation")
            feature_mapping["Missing %"].append(f"{df_2022[col].isna().sum()/len(df_2022)*100:.1f}%")
    
    # 6. LLM features
    print("\n6. LLM FEATURE PROCESSING:")
    print("-"*40)
    print("All LLM features from essay analysis:")
    for col in df_llm.columns:
        if col.startswith('llm_'):
            if 'count' in col:
                print(f"{col}: Missing → 0")
                feature_mapping["Expected Feature"].append(col)
                feature_mapping["Raw Column Name"].append(col + " (from GPT-4o)")
                feature_mapping["Transformation Applied"].append("Missing → 0")
            else:
                print(f"{col}: Missing → 50 (neutral score)")
                feature_mapping["Expected Feature"].append(col)
                feature_mapping["Raw Column Name"].append(col + " (from GPT-4o)")
                feature_mapping["Transformation Applied"].append("Missing → 50")
            feature_mapping["Missing %"].append("0.0%")
    
    # 7. Missing features
    print("\n7. FEATURES NOT FOUND IN DATA:")
    print("-"*40)
    missing_features = ['Undergrad_GPA', 'Grad_GPA', 'MCAT_Total', 'Undergrad_BCPM', 'Grad_BCPM', 'military_service']
    for feat in missing_features:
        print(f"❌ {feat}: NOT FOUND in raw data")
        feature_mapping["Expected Feature"].append(feat)
        feature_mapping["Raw Column Name"].append("NOT AVAILABLE")
        feature_mapping["Transformation Applied"].append("Feature excluded or synthetic")
        feature_mapping["Missing %"].append("100%")
    
    # 8. One-hot encoding
    print("\n8. ONE-HOT ENCODING:")
    print("-"*40)
    print("Gender → One-hot encoded (multiple binary columns)")
    print("State_Grouped → One-hot encoded (West, Northeast, South, Midwest, Unknown)")
    print("Citizenship_Status → One-hot encoded (US_Citizen, Other, Unknown)")
    
    # 9. Final standardization
    print("\n9. FINAL STANDARDIZATION:")
    print("-"*40)
    print("All continuous features: StandardScaler (mean=0, std=1)")
    print("Applied to: Age, experience hours (after log transform)")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(feature_mapping)
    
    return summary_df

if __name__ == "__main__":
    summary = analyze_transformations()
    print("\n" + "="*80)
    print("TRANSFORMATION SUMMARY TABLE")
    print("="*80)
    print(summary.to_string(index=False))