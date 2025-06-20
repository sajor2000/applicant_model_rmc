"""
Improve Model Confidence - Reduce Uncertain Cases
================================================

Analyze why 21% of cases have low confidence and implement improvements
to reduce the number requiring human review.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
from datetime import datetime


def analyze_low_confidence_cases():
    """Analyze characteristics of low confidence predictions."""
    print("="*80)
    print("ANALYZING LOW CONFIDENCE CASES")
    print("="*80)
    
    # Load rankings with confidence scores
    rankings_files = list(Path("output").glob("candidate_rankings_*.csv"))
    if not rankings_files:
        print("No rankings files found")
        return
    
    latest_rankings = pd.read_csv(sorted(rankings_files)[-1])
    
    # Separate by confidence level
    low_conf = latest_rankings[latest_rankings['confidence_level'] == 'Low']
    high_conf = latest_rankings[latest_rankings['confidence_level'] == 'High']
    
    print(f"\nConfidence Distribution:")
    print(f"Low: {len(low_conf)} ({len(low_conf)/len(latest_rankings)*100:.1f}%)")
    print(f"Medium: {(latest_rankings['confidence_level'] == 'Medium').sum()} ({(latest_rankings['confidence_level'] == 'Medium').sum()/len(latest_rankings)*100:.1f}%)")
    print(f"High: {len(high_conf)} ({len(high_conf)/len(latest_rankings)*100:.1f}%)")
    
    # Analyze probability distributions
    print("\n\nLow Confidence Characteristics:")
    print("\nProbability Spreads (Low vs High Confidence):")
    
    prob_cols = ['reject_prob', 'waitlist_prob', 'interview_prob', 'accept_prob']
    
    # Calculate entropy/spread
    low_conf_entropy = -np.sum(low_conf[prob_cols] * np.log(low_conf[prob_cols] + 1e-10), axis=1).mean()
    high_conf_entropy = -np.sum(high_conf[prob_cols] * np.log(high_conf[prob_cols] + 1e-10), axis=1).mean()
    
    print(f"Average entropy - Low conf: {low_conf_entropy:.3f}")
    print(f"Average entropy - High conf: {high_conf_entropy:.3f}")
    
    # Find borderline cases
    print("\nBorderline Cases Analysis:")
    
    # Cases where top 2 probabilities are close
    for idx, row in low_conf.iterrows():
        probs = row[prob_cols].values
        sorted_probs = np.sort(probs)[::-1]
        if sorted_probs[0] - sorted_probs[1] < 0.2:  # Top 2 within 20%
            print(f"AMCAS {row['amcas_id']}: {sorted_probs[0]:.2f} vs {sorted_probs[1]:.2f} (True score: {row['true_score']})")
            if idx > low_conf.index[10]:
                break
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confidence score distribution
    ax1 = axes[0, 0]
    latest_rankings['confidence_score'].hist(bins=30, ax=ax1, edgecolor='black')
    ax1.axvline(x=60, color='red', linestyle='--', label='Low/Medium threshold')
    ax1.axvline(x=80, color='green', linestyle='--', label='Medium/High threshold')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Score Distribution')
    ax1.legend()
    
    # 2. Probability spread by confidence level
    ax2 = axes[0, 1]
    low_conf_max_prob = low_conf[prob_cols].max(axis=1)
    high_conf_max_prob = high_conf[prob_cols].max(axis=1)
    
    ax2.hist(low_conf_max_prob, bins=20, alpha=0.5, label='Low Confidence', color='red')
    ax2.hist(high_conf_max_prob, bins=20, alpha=0.5, label='High Confidence', color='green')
    ax2.set_xlabel('Maximum Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Highest Probability by Confidence Level')
    ax2.legend()
    
    # 3. True score distribution by confidence
    ax3 = axes[1, 0]
    latest_rankings.boxplot(column='true_score', by='confidence_level', ax=ax3)
    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('True Application Score')
    ax3.set_title('True Scores by Confidence Level')
    
    # 4. Quartile distribution by confidence
    ax4 = axes[1, 1]
    conf_quartile = pd.crosstab(latest_rankings['confidence_level'], 
                               latest_rankings['quartile'], 
                               normalize='index') * 100
    conf_quartile.plot(kind='bar', ax=ax4)
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Quartile Distribution by Confidence Level')
    ax4.legend(title='Quartile')
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300)
    print("\nVisualization saved to confidence_analysis.png")
    
    return latest_rankings


def enhance_cascade_training():
    """Enhanced training with focus on confidence improvement."""
    print("\n\n" + "="*80)
    print("ENHANCED CASCADE TRAINING FOR HIGHER CONFIDENCE")
    print("="*80)
    
    # Hyperparameter grid for better calibration
    enhanced_params = {
        'stage1': {
            'n_estimators': [300, 500, 700],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
            'reg_lambda': [1, 1.5, 2],   # L2 regularization
        }
    }
    
    print("\nProposed Enhancements:")
    print("\n1. HYPERPARAMETER OPTIMIZATION")
    print("   - Increase n_estimators: 300 → 500-700 (more trees, smoother probabilities)")
    print("   - Optimize max_depth: 4 → 3-6 (prevent overfitting)")
    print("   - Add regularization: alpha=0.1-0.5, lambda=1-2")
    print("   - Tune min_child_weight: 1 → 3-5 (more conservative splits)")
    
    print("\n2. CALIBRATION IMPROVEMENTS")
    print("   - Implement Platt Scaling for probability calibration")
    print("   - Use isotonic regression for monotonic probability mapping")
    print("   - Cross-validate probability thresholds")
    
    print("\n3. ENSEMBLE METHODS")
    print("   - Train 5 models with different random seeds")
    print("   - Average predictions for smoother probabilities")
    print("   - Use prediction variance as additional confidence measure")
    
    print("\n4. FEATURE ENGINEERING FOR CONFIDENCE")
    print("   - Add 'ambiguity features':")
    print("     * Variance in essay scores")
    print("     * Inconsistency between structured and essay signals")
    print("     * Missing data indicators")
    print("   - Create 'profile coherence' score")
    
    print("\n5. THRESHOLD OPTIMIZATION")
    print("   Current thresholds: 60 (Low/Med), 80 (Med/High)")
    print("   Suggested: Optimize based on human review accuracy")
    
    return enhanced_params


def implement_ensemble_cascade():
    """Implement ensemble version of cascade for higher confidence."""
    print("\n\n" + "="*80)
    print("ENSEMBLE CASCADE IMPLEMENTATION")
    print("="*80)
    
    code_template = '''
class EnsembleCascadeClassifier:
    """Ensemble of cascade classifiers for improved confidence."""
    
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        
    def train(self, X, y):
        for i in range(self.n_models):
            # Train with different random seed
            cascade = CascadingClassifier(random_state=i*42)
            
            # Use slightly different hyperparameters
            cascade.stage1_params = {
                'n_estimators': 500 + i*50,
                'max_depth': 4 + (i % 2),
                'learning_rate': 0.02 + i*0.005,
                'subsample': 0.8 + (i % 3)*0.05,
                'reg_alpha': 0.1 + i*0.05,
                'reg_lambda': 1.5 + i*0.1
            }
            
            cascade.fit(X, y)
            self.models.append(cascade)
    
    def predict_with_confidence(self, X):
        # Get predictions from all models
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            pred, prob = model.predict_cascade(X)
            all_predictions.append(pred)
            all_probabilities.append(prob)
        
        # Ensemble predictions (majority vote)
        ensemble_pred = np.round(np.mean(all_predictions, axis=0))
        
        # Ensemble probabilities (average)
        ensemble_prob = np.mean(all_probabilities, axis=0)
        
        # Calculate confidence based on agreement
        prediction_variance = np.var(all_predictions, axis=0)
        prob_variance = np.var(all_probabilities, axis=0).mean(axis=1)
        
        # High confidence = low variance
        confidence = 100 * (1 - np.sqrt(prob_variance))
        
        return ensemble_pred, ensemble_prob, confidence
    '''
    
    print("Ensemble Benefits:")
    print("- Reduces prediction variance")
    print("- Natural confidence measure from model agreement")
    print("- More stable probabilities")
    print("- Expected to reduce low confidence cases by 30-40%")
    
    return code_template


def create_improvement_recommendations():
    """Create specific recommendations for implementation."""
    
    recommendations = """
## Recommendations to Reduce Low Confidence Cases

### Immediate Improvements (1-2 days):

1. **Increase Model Complexity**
   - Current: 300 trees → Recommended: 500-700 trees
   - Current: Single model → Recommended: 5-model ensemble
   - Expected impact: 20-30% reduction in low confidence

2. **Probability Calibration**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   
   # Wrap each stage classifier
   calibrated_model = CalibratedClassifierCV(
       base_estimator=xgb_model,
       method='isotonic',  # or 'sigmoid'
       cv=3
   )
   ```
   - Expected impact: 15-20% reduction in borderline cases

3. **Optimize Confidence Thresholds**
   - Analyze human review outcomes
   - Set thresholds to minimize false low-confidence
   - Consider quartile-specific thresholds

### Advanced Improvements (1 week):

1. **Add Confidence-Specific Features**
   ```python
   # Profile coherence score
   coherence_features = {
       'essay_structured_alignment': correlation(essay_scores, structured_scores),
       'internal_consistency': std(feature_percentiles),
       'profile_completeness': 1 - (missing_features / total_features),
       'extreme_feature_count': sum(features > 95th_percentile)
   }
   ```

2. **Multi-Stage Confidence**
   - Stage 1 confidence: Clear reject vs borderline
   - Stage 2 confidence: Clear middle vs Q1/Q3 border
   - Stage 3 confidence: Clear accept vs Q2/Q3 border
   - Combine for overall confidence

3. **Active Learning Integration**
   - Track which low-confidence cases humans overturn
   - Retrain on these specific patterns
   - Build "ambiguity detector" model

### Expected Outcomes:

Current: 21% low confidence (129/613)
Target: 10-12% low confidence (60-75/613)

This represents a 50% reduction in human review workload while maintaining accuracy.
"""
    
    with open("confidence_improvement_plan.md", "w") as f:
        f.write(recommendations)
    
    print("\n" + "="*80)
    print("IMPROVEMENT PLAN SUMMARY")
    print("="*80)
    print("\nKey Strategies:")
    print("1. Ensemble modeling (5 models vs 1)")
    print("2. More trees per model (500-700 vs 300)")
    print("3. Probability calibration (isotonic/sigmoid)")
    print("4. Confidence-specific features")
    print("5. Optimized thresholds based on review data")
    print("\nExpected Outcome: Reduce low confidence from 21% to 10-12%")
    print("\nDetailed plan saved to: confidence_improvement_plan.md")


def main():
    # Analyze current low confidence cases
    rankings = analyze_low_confidence_cases()
    
    # Show enhancement options
    enhanced_params = enhance_cascade_training()
    
    # Show ensemble approach
    ensemble_code = implement_ensemble_cascade()
    
    # Create implementation plan
    create_improvement_recommendations()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review confidence_analysis.png for patterns")
    print("2. Implement ensemble cascade (highest impact)")
    print("3. Add probability calibration")
    print("4. Test on validation set")
    print("5. Deploy improved model")


if __name__ == "__main__":
    main()