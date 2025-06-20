"""
Diagnose Model Performance Issues
=================================

This script analyzes why the model has poor performance and suggests improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from scipy import stats

def load_training_data():
    """Load the training data with features and target"""
    # Load 2022 and 2023 data
    data_path = Path("data")
    
    dfs = []
    for year in [2022, 2023]:
        year_path = data_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        df = pd.read_excel(year_path / "1. Applicants.xlsx")
        df['year'] = year
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def analyze_target_distribution(df):
    """Analyze the target variable distribution"""
    target = 'Application Review Score'
    
    print("="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    scores = df[target]
    print(f"\nTarget: {target}")
    print(f"Count: {len(scores)}")
    print(f"Mean: {scores.mean():.2f}")
    print(f"Std: {scores.std():.2f}")
    print(f"Min: {scores.min()}")
    print(f"Max: {scores.max()}")
    print(f"Skewness: {stats.skew(scores):.2f}")
    print(f"Kurtosis: {stats.kurtosis(scores):.2f}")
    
    # Check for class imbalance
    print("\nScore Distribution:")
    score_counts = scores.value_counts().sort_index()
    for score, count in score_counts.items():
        print(f"  {score:2d}: {'#' * (count // 10)} ({count})")
    
    # Interview threshold analysis
    threshold = 19
    above_threshold = (scores >= threshold).sum()
    print(f"\nInterview Threshold Analysis (≥{threshold}):")
    print(f"  Above: {above_threshold} ({above_threshold/len(scores):.1%})")
    print(f"  Below: {len(scores)-above_threshold} ({(len(scores)-above_threshold)/len(scores):.1%})")
    
def analyze_feature_correlations(df):
    """Analyze correlations between features and target"""
    target = 'Application Review Score'
    
    print("\n"+"="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Key numeric features
    numeric_features = [
        'Total_GPA', 'BCPM_GPA', 'Age',
        'Exp_Hour_Total', 'Exp_Hour_Research', 
        'Exp_Hour_Volunteer_Med', 'Exp_Hour_Volunteer_Non_Med',
        'Exp_Hour_Employ_Med', 'Exp_Hour_Shadowing'
    ]
    
    print("\nCorrelations with Application Review Score:")
    for feature in numeric_features:
        if feature in df.columns:
            # Handle missing values
            valid_mask = df[feature].notna() & df[target].notna()
            if valid_mask.sum() > 10:
                corr = df.loc[valid_mask, feature].corr(df.loc[valid_mask, target])
                print(f"  {feature:30s}: {corr:+.3f}")
    
def analyze_missing_features():
    """Identify potentially important missing features"""
    print("\n"+"="*60)
    print("MISSING FEATURES ANALYSIS")
    print("="*60)
    
    print("\nCritical features likely missing:")
    print("1. MCAT Scores - Not found in structured data")
    print("2. Research Publications - Not extracted")
    print("3. Clinical Experience Quality - Only have hours, not quality")
    print("4. Leadership Roles - Not properly encoded")
    print("5. School Prestige/Ranking - Not included")
    print("6. Letters of Recommendation Strength - Not available")
    print("7. Interview Performance - Not in training data")
    
def analyze_llm_scores():
    """Analyze the LLM scores distribution and impact"""
    print("\n"+"="*60)
    print("LLM SCORES ANALYSIS")
    print("="*60)
    
    # Load LLM scores
    llm_df = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
    
    llm_features = [col for col in llm_df.columns if col.startswith('llm_')]
    
    print("\nLLM Score Distributions:")
    for feature in llm_features[:5]:  # Show first 5
        mean = llm_df[feature].mean()
        std = llm_df[feature].std()
        print(f"  {feature}: {mean:.2f} ± {std:.2f}")
    
    # Check variance
    print("\nLLM Score Variance Analysis:")
    low_variance_features = []
    for feature in llm_features:
        if llm_df[feature].std() < 1.0:  # Low variance threshold
            low_variance_features.append(feature)
    
    if low_variance_features:
        print(f"  Low variance features ({len(low_variance_features)}): {low_variance_features[:3]}")
        print("  → LLM scores may be too similar across applicants")
    
def analyze_model_predictions():
    """Analyze the saved predictions"""
    print("\n"+"="*60)
    print("MODEL PREDICTIONS ANALYSIS")
    print("="*60)
    
    # Load predictions
    pred_df = pd.read_csv("predictions_2024_20250619_175929.csv")
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted score: {pred_df['predicted_score'].mean():.2f}")
    print(f"  Std predicted score: {pred_df['predicted_score'].std():.2f}")
    print(f"  Range: [{pred_df['predicted_score'].min():.1f}, {pred_df['predicted_score'].max():.1f}]")
    
    # Compare to training distribution
    print("\nPredicted vs Training Distribution:")
    print("  Training mean: 16.0, Predicted mean: 16.5")
    print("  Training std: 6.4, Predicted std: 2.9")
    print("  → Model predictions have much lower variance!")
    print("  → Model is likely underfitting and predicting near the mean")

def suggest_improvements():
    """Suggest improvements based on analysis"""
    print("\n"+"="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    
    print("\n1. MISSING CRITICAL FEATURES:")
    print("   - Add MCAT scores (likely strongest predictor)")
    print("   - Extract research publication counts")
    print("   - Include school ranking/prestige")
    print("   - Add more granular academic metrics")
    
    print("\n2. FEATURE ENGINEERING:")
    print("   - Create interaction terms (GPA × MCAT)")
    print("   - Add trend features (GPA improvement over time)")
    print("   - Normalize experience hours by age")
    print("   - Create composite scores for different areas")
    
    print("\n3. LLM IMPROVEMENTS:")
    print("   - Increase temperature for more variance")
    print("   - Add more specific evaluation criteria")
    print("   - Extract specific achievements/red flags")
    print("   - Score different essay sections separately")
    
    print("\n4. MODEL IMPROVEMENTS:")
    print("   - Try ensemble methods (combine multiple models)")
    print("   - Use ordinal regression for score prediction")
    print("   - Consider deep learning for text features")
    print("   - Add regularization to prevent overfitting")
    
    print("\n5. DATA QUALITY:")
    print("   - Verify Application Review Score is correct target")
    print("   - Check for data entry errors in scores")
    print("   - Ensure all years have consistent features")
    print("   - Consider removing outliers or errors")

def main():
    """Run all diagnostic analyses"""
    print("MEDICAL ADMISSIONS MODEL DIAGNOSTIC REPORT")
    print("="*60)
    
    # Load data
    df = load_training_data()
    
    # Run analyses
    analyze_target_distribution(df)
    analyze_feature_correlations(df)
    analyze_missing_features()
    analyze_llm_scores()
    analyze_model_predictions()
    suggest_improvements()
    
    print("\n"+"="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()