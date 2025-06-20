"""
Analyze Different Bucketing Strategies for Medical Admissions
============================================================

This script explores optimal ways to create 4 buckets from reviewer scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_scores():
    """Load all Application Review Scores"""
    data_path = Path("data")
    
    all_scores = []
    for year in [2022, 2023]:
        year_path = data_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        df = pd.read_excel(year_path / "1. Applicants.xlsx")
        if 'Application Review Score' in df.columns:
            scores = df['Application Review Score'].values
            all_scores.extend(scores)
    
    return np.array(all_scores)

def analyze_score_distribution(scores):
    """Analyze the distribution of scores"""
    print("="*60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Basic stats
    print(f"\nTotal applicants: {len(scores)}")
    print(f"Score range: [{scores.min()}, {scores.max()}]")
    print(f"Mean: {scores.mean():.2f}, Median: {np.median(scores):.1f}")
    
    # Unique values
    unique_scores, counts = np.unique(scores, return_counts=True)
    print(f"\nUnique scores: {len(unique_scores)}")
    print("Distribution:")
    for score, count in zip(unique_scores, counts):
        bar = '#' * (count // 10)
        print(f"  {score:2.0f}: {bar:20s} ({count:3d}, {count/len(scores)*100:4.1f}%)")
    
    return unique_scores, counts

def calculate_bucket_strategies(scores):
    """Calculate different bucketing strategies"""
    print("\n"+"="*60)
    print("BUCKETING STRATEGIES")
    print("="*60)
    
    strategies = {}
    
    # Strategy 1: Quartiles (Equal size buckets)
    q1, q2, q3 = np.percentile(scores, [25, 50, 75])
    strategies['quartiles'] = {
        'name': 'Quartiles (Equal Size)',
        'boundaries': [scores.min(), q1, q2, q3, scores.max()],
        'labels': ['Bottom 25%', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Top 25%']
    }
    
    # Strategy 2: Natural Breaks (based on score clusters)
    # Looking at the distribution: 0-9, 11-15, 17-21, 23-25
    strategies['natural'] = {
        'name': 'Natural Breaks',
        'boundaries': [0, 10, 16, 22, 26],
        'labels': ['Reject (0-9)', 'Waitlist (11-15)', 'Interview (17-21)', 'Accept (23-25)']
    }
    
    # Strategy 3: Interview-focused (emphasize the 19 threshold)
    strategies['interview'] = {
        'name': 'Interview-Focused',
        'boundaries': [0, 13, 18, 21, 26],
        'labels': ['Reject (0-12)', 'Maybe (13-17)', 'Interview (18-20)', 'Strong (21-25)']
    }
    
    # Strategy 4: Top-heavy (identify exceptional candidates)
    strategies['top_heavy'] = {
        'name': 'Top-Heavy Selection',
        'boundaries': [0, 15, 19, 23, 26],
        'labels': ['No (0-14)', 'Maybe (15-18)', 'Yes (19-22)', 'Exceptional (23-25)']
    }
    
    # Strategy 5: Data-driven (using Jenks natural breaks)
    try:
        from jenkspy import jenks_breaks
        breaks = jenks_breaks(scores, n_classes=4)
        strategies['jenks'] = {
            'name': 'Jenks Natural Breaks',
            'boundaries': breaks,
            'labels': [f'Tier {i+1}' for i in range(4)]
        }
    except ImportError:
        print("Note: Install jenkspy for Jenks natural breaks algorithm")
    
    return strategies

def evaluate_strategy(scores, strategy):
    """Evaluate a bucketing strategy"""
    boundaries = strategy['boundaries']
    labels = strategy['labels']
    
    # Create buckets
    buckets = np.digitize(scores, boundaries[1:-1])
    
    # Count distribution
    print(f"\n{strategy['name']}:")
    print(f"Boundaries: {boundaries}")
    
    for i, label in enumerate(labels):
        count = np.sum(buckets == i)
        pct = count / len(scores) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Calculate metrics
    # Variance within buckets (lower is better)
    within_var = 0
    for i in range(len(labels)):
        bucket_scores = scores[buckets == i]
        if len(bucket_scores) > 0:
            within_var += np.var(bucket_scores) * len(bucket_scores)
    within_var /= len(scores)
    
    # Separation between buckets (higher is better)
    bucket_means = []
    for i in range(len(labels)):
        bucket_scores = scores[buckets == i]
        if len(bucket_scores) > 0:
            bucket_means.append(np.mean(bucket_scores))
    
    if len(bucket_means) > 1:
        separation = np.min(np.diff(sorted(bucket_means)))
    else:
        separation = 0
    
    print(f"  Within-bucket variance: {within_var:.2f}")
    print(f"  Min separation between means: {separation:.2f}")
    
    return buckets, within_var, separation

def recommend_modeling_approach(scores, strategies):
    """Recommend the best modeling approach"""
    print("\n"+"="*60)
    print("MODELING APPROACH RECOMMENDATIONS")
    print("="*60)
    
    # Evaluate all strategies
    results = {}
    for key, strategy in strategies.items():
        buckets, within_var, separation = evaluate_strategy(scores, strategy)
        results[key] = {
            'within_var': within_var,
            'separation': separation,
            'score': separation / (within_var + 1)  # Combined metric
        }
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['score'])[0]
    
    print(f"\n✓ RECOMMENDED BUCKETING: {strategies[best_strategy]['name']}")
    print(f"  Boundaries: {strategies[best_strategy]['boundaries']}")
    
    print("\n✓ RECOMMENDED MODELING APPROACH:")
    print("\n1. PRIMARY: Ordinal Regression")
    print("   - Respects natural ordering of buckets")
    print("   - Better calibrated probabilities")
    print("   - Can output both bucket and confidence")
    
    print("\n2. ENSEMBLE APPROACH:")
    print("   a) XGBoost Ordinal Regression (main model)")
    print("   b) Neural Network with ordinal output layer")
    print("   c) Random Forest with custom ordinal split criterion")
    print("   → Combine predictions using weighted average")
    
    print("\n3. TWO-STAGE APPROACH:")
    print("   Stage 1: Binary classification (Interview Yes/No at score 19)")
    print("   Stage 2: Within each group, classify into 2 sub-buckets")
    print("   → More balanced classes at each stage")
    
    print("\n4. FEATURES TO EMPHASIZE:")
    print("   - Create bucket-specific features")
    print("   - Add polynomial features for top performers")
    print("   - Use different feature weights per bucket")
    
    print("\n5. LOSS FUNCTION OPTIMIZATION:")
    print("   - Use weighted ordinal loss")
    print("   - Penalize boundary errors less than distant errors")
    print("   - Custom metric: Quadratic Weighted Kappa")
    
    return strategies[best_strategy]

def create_implementation_plan():
    """Create detailed implementation plan"""
    print("\n"+"="*60)
    print("IMPLEMENTATION PLAN")
    print("="*60)
    
    print("\n📋 PHASE 1: Data Preparation")
    print("1. Create balanced buckets using recommended boundaries")
    print("2. Generate bucket-specific features")
    print("3. Implement SMOTE for minority buckets")
    print("4. Create stratified train/val/test splits")
    
    print("\n📋 PHASE 2: Model Development")
    print("1. Implement XGBoost with custom ordinal objective")
    print("2. Create neural network with ordinal activation")
    print("3. Develop evaluation metrics (QWK, bucket accuracy)")
    print("4. Implement confidence calibration")
    
    print("\n📋 PHASE 3: Feature Engineering 2.0")
    print("1. Bucket-aware feature scaling")
    print("2. Interaction terms for boundary cases")
    print("3. Ensemble LLM scores with different prompts")
    print("4. Add meta-features from model disagreement")
    
    print("\n📋 PHASE 4: Production Pipeline")
    print("1. Multi-model ensemble prediction")
    print("2. Confidence scores for each bucket")
    print("3. Explanation generation")
    print("4. Fairness monitoring")

def main():
    """Run the analysis"""
    # Load scores
    scores = load_scores()
    
    # Analyze distribution
    unique_scores, counts = analyze_score_distribution(scores)
    
    # Calculate bucketing strategies
    strategies = calculate_bucket_strategies(scores)
    
    # Recommend approach
    best_strategy = recommend_modeling_approach(scores, strategies)
    
    # Implementation plan
    create_implementation_plan()
    
    print("\n"+"="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()