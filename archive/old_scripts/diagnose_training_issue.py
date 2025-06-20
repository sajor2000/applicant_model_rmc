"""
Diagnose Training Issues
=======================

Check data distribution and feature quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data():
    """Analyze the filtered data to understand issues."""
    
    print("="*80)
    print("DATA DIAGNOSIS")
    print("="*80)
    
    # Load filtered data
    df_2022 = pd.read_excel("data_filtered/2022_filtered_applicants.xlsx")
    df_2023 = pd.read_excel("data_filtered/2023_filtered_applicants.xlsx")
    df_2024 = pd.read_excel("data_filtered/2024_filtered_applicants.xlsx")
    
    # Combine training data
    df_train = pd.concat([df_2022, df_2023], ignore_index=True)
    
    print("\n1. Data Shape:")
    print(f"   2022: {df_2022.shape}")
    print(f"   2023: {df_2023.shape}")
    print(f"   2024: {df_2024.shape}")
    print(f"   Combined training: {df_train.shape}")
    
    # Check target distribution
    print("\n2. Target Distribution (Application Review Score):")
    
    # Define buckets
    bucket_boundaries = [0, 10, 16, 22, 26]
    bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
    
    for name, df in [("2022", df_2022), ("2023", df_2023), ("2024", df_2024), ("Train", df_train)]:
        scores = df['application_review_score']
        buckets = pd.cut(scores, bins=bucket_boundaries, labels=bucket_names, include_lowest=True)
        print(f"\n   {name}:")
        print(f"   Score range: {scores.min():.0f} - {scores.max():.0f}")
        print(f"   Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")
        print("   Bucket distribution:")
        for bucket in bucket_names:
            count = (buckets == bucket).sum()
            pct = count / len(buckets) * 100
            print(f"     {bucket}: {count} ({pct:.1f}%)")
    
    # Check key features
    print("\n3. Key Feature Analysis:")
    
    key_features = ['service_rating_numerical', 'healthcare_total_hours', 
                   'exp_hour_total', 'age', 'ses_value']
    
    for feature in key_features:
        if feature in df_train.columns:
            print(f"\n   {feature}:")
            print(f"   - Non-null: {df_train[feature].notna().sum()} ({df_train[feature].notna().sum()/len(df_train)*100:.1f}%)")
            print(f"   - Mean: {df_train[feature].mean():.2f}")
            print(f"   - Std: {df_train[feature].std():.2f}")
            print(f"   - Range: {df_train[feature].min():.2f} - {df_train[feature].max():.2f}")
            
            # Correlation with target
            corr = df_train[[feature, 'application_review_score']].corr().iloc[0, 1]
            print(f"   - Correlation with target: {corr:.3f}")
    
    # Check for data leakage
    print("\n4. Checking for potential issues:")
    
    # Check if any features have perfect correlation with target
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    exclude_cols = ['application_review_score', 'amcas_id', 'appl_year']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    correlations = df_train[feature_cols + ['application_review_score']].corr()['application_review_score'].drop('application_review_score')
    high_corr = correlations[correlations.abs() > 0.7]
    
    if len(high_corr) > 0:
        print("   HIGH CORRELATION FEATURES (>0.7):")
        for feat, corr in high_corr.items():
            print(f"   - {feat}: {corr:.3f}")
    else:
        print("   No features with high correlation (>0.7) to target")
    
    # Check feature variance
    print("\n   Low variance features (<0.01):")
    low_var_count = 0
    for col in feature_cols:
        if df_train[col].notna().any():
            var = df_train[col].var()
            if var < 0.01:
                print(f"   - {col}: var={var:.4f}")
                low_var_count += 1
    
    if low_var_count == 0:
        print("   No low variance features found")
    
    # Check class imbalance
    print("\n5. Class Balance Check:")
    y_train = df_train['application_review_score'].values
    y_buckets = np.digitize(y_train, bucket_boundaries[1:-1]) - 1
    
    for i, name in enumerate(bucket_names):
        count = (y_buckets == i).sum()
        pct = count / len(y_buckets) * 100
        print(f"   {name} (class {i}): {count} ({pct:.1f}%)")
    
    # Save visualizations
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Score distribution
    ax1 = axes[0, 0]
    df_train['application_review_score'].hist(bins=26, ax=ax1, edgecolor='black')
    ax1.set_title('Training Score Distribution')
    ax1.set_xlabel('Application Review Score')
    ax1.set_ylabel('Count')
    
    # Service rating vs score
    ax2 = axes[0, 1]
    if 'service_rating_numerical' in df_train.columns:
        ax2.scatter(df_train['service_rating_numerical'], 
                   df_train['application_review_score'], 
                   alpha=0.5)
        ax2.set_xlabel('Service Rating')
        ax2.set_ylabel('Application Review Score')
        ax2.set_title('Service Rating vs Score')
    
    # Feature correlations
    ax3 = axes[1, 0]
    top_corr_features = correlations.abs().sort_values(ascending=False).head(10)
    top_corr_features.plot(kind='barh', ax=ax3)
    ax3.set_title('Top 10 Feature Correlations with Target')
    ax3.set_xlabel('Absolute Correlation')
    
    # Bucket distribution comparison
    ax4 = axes[1, 1]
    bucket_data = []
    for name, df in [("2022", df_2022), ("2023", df_2023), ("2024", df_2024)]:
        scores = df['application_review_score']
        buckets = pd.cut(scores, bins=bucket_boundaries, labels=bucket_names, include_lowest=True)
        bucket_counts = buckets.value_counts(normalize=True).sort_index()
        bucket_data.append(bucket_counts)
    
    bucket_df = pd.DataFrame(bucket_data, index=["2022", "2023", "2024"]).T
    bucket_df.plot(kind='bar', ax=ax4)
    ax4.set_title('Bucket Distribution by Year')
    ax4.set_ylabel('Proportion')
    ax4.legend(title='Year')
    
    plt.tight_layout()
    plt.savefig('data_diagnosis.png', dpi=300, bbox_inches='tight')
    print("   Saved visualizations to data_diagnosis.png")
    
    # Load and check LLM scores
    print("\n7. LLM Score Analysis:")
    llm_train = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
    llm_train = llm_train.rename(columns={'AMCAS_ID_original': 'amcas_id'})
    
    # Merge with training data
    df_with_llm = df_train.merge(llm_train, on='amcas_id', how='left')
    
    llm_cols = [col for col in llm_train.columns if col.startswith('llm_')]
    print(f"   Found {len(llm_cols)} LLM features")
    print(f"   Merged successfully: {df_with_llm[llm_cols[0]].notna().sum()}/{len(df_train)}")
    
    # Check LLM feature correlations
    if llm_cols:
        llm_corrs = df_with_llm[llm_cols + ['application_review_score']].corr()['application_review_score'].drop('application_review_score')
        print("\n   LLM feature correlations with target:")
        for feat, corr in llm_corrs.sort_values(ascending=False).head(5).items():
            print(f"   - {feat}: {corr:.3f}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    analyze_data()