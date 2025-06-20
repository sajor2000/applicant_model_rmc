#!/usr/bin/env python3
"""
Analyze data quality and missingness in the training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

def analyze_missingness():
    """Analyze missing data patterns in training data"""
    
    # Load 2022 and 2023 data
    all_data = []
    
    for year in ['2022', '2023']:
        print(f"\nAnalyzing {year} data...")
        
        # Load structured data
        df_struct = pd.read_excel(f"data/{year} Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
        
        # Load LLM scores
        df_llm = pd.read_csv(f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv")
        
        # Standardize AMCAS ID
        if 'Amcas_ID' in df_struct.columns:
            df_struct['AMCAS ID'] = df_struct['Amcas_ID'].astype(str)
        df_llm['AMCAS ID'] = df_llm['AMCAS ID'].astype(str)
        
        # Merge
        df_merged = pd.merge(df_struct, df_llm, on='AMCAS ID', how='inner')
        all_data.append(df_merged)
    
    # Combine all training data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal training records: {len(combined_df)}")
    
    # Define feature groups as used in FeatureEngineer
    continuous_features = [
        'Age', 'Undergrad_GPA', 'Grad_GPA', 'MCAT_Total',
        'Undergrad_BCPM', 'Grad_BCPM'
    ]
    
    categorical_features = [
        'Gender', 'First_Generation_Ind', 'Disadvantanged_Ind',
        'military_service', 'State', 'Citizenship'
    ]
    
    experience_features = [
        'Exp_Hour_Research', 'Exp_Hour_Volunteer_Med', 
        'Exp_Hour_Volunteer_Non_Med', 'Comm_Service_Total_Hours',
        'HealthCare_Total_Hours'
    ]
    
    llm_features = [col for col in combined_df.columns if col.startswith('llm_')]
    
    # Analyze missingness by feature group
    print("\n" + "="*60)
    print("MISSINGNESS ANALYSIS BY FEATURE GROUP")
    print("="*60)
    
    results = {}
    
    # 1. Continuous features
    print("\n1. CONTINUOUS FEATURES (Academic Metrics):")
    print("-"*40)
    for feat in continuous_features:
        if feat in combined_df.columns:
            missing_count = combined_df[feat].isna().sum()
            missing_pct = (missing_count / len(combined_df)) * 100
            results[feat] = missing_pct
            print(f"{feat:20s}: {missing_pct:5.1f}% missing ({missing_count} records)")
        else:
            # Check if we need to derive it
            if feat == 'Age' and 'Date of Birth' in combined_df.columns:
                missing_count = combined_df['Date of Birth'].isna().sum()
                missing_pct = (missing_count / len(combined_df)) * 100
                results['Date of Birth (→Age)'] = missing_pct
                print(f"Date of Birth (→Age): {missing_pct:5.1f}% missing ({missing_count} records)")
    
    # 2. Categorical features
    print("\n2. CATEGORICAL FEATURES (Demographics):")
    print("-"*40)
    for feat in categorical_features:
        if feat in combined_df.columns:
            missing_count = combined_df[feat].isna().sum()
            missing_pct = (missing_count / len(combined_df)) * 100
            results[feat] = missing_pct
            print(f"{feat:20s}: {missing_pct:5.1f}% missing ({missing_count} records)")
    
    # 3. Experience features
    print("\n3. EXPERIENCE FEATURES (Hours):")
    print("-"*40)
    for feat in experience_features:
        if feat in combined_df.columns:
            missing_count = combined_df[feat].isna().sum()
            missing_pct = (missing_count / len(combined_df)) * 100
            results[feat] = missing_pct
            print(f"{feat:25s}: {missing_pct:5.1f}% missing ({missing_count} records)")
    
    # 4. LLM features
    print("\n4. LLM FEATURES (Essay Analysis):")
    print("-"*40)
    for feat in llm_features:
        missing_count = combined_df[feat].isna().sum()
        missing_pct = (missing_count / len(combined_df)) * 100
        results[feat] = missing_pct
        print(f"{feat:30s}: {missing_pct:5.1f}% missing ({missing_count} records)")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    all_missing_pcts = list(results.values())
    print(f"Average missingness across all features: {np.mean(all_missing_pcts):.1f}%")
    print(f"Maximum missingness: {max(all_missing_pcts):.1f}%")
    print(f"Minimum missingness: {min(all_missing_pcts):.1f}%")
    print(f"Features with >10% missing: {sum(1 for x in all_missing_pcts if x > 10)}")
    print(f"Features with >20% missing: {sum(1 for x in all_missing_pcts if x > 20)}")
    
    # Raw column mapping
    print("\n" + "="*60)
    print("RAW COLUMN NAMES IN DATASET")
    print("="*60)
    
    # Show first 50 columns
    all_cols = list(combined_df.columns)
    print(f"Total columns in raw data: {len(all_cols)}")
    print("\nFirst 50 columns:")
    for i, col in enumerate(all_cols[:50]):
        print(f"{i+1:3d}. {col}")
    
    return results, combined_df

if __name__ == "__main__":
    results, df = analyze_missingness()