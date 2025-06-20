"""
Comprehensive Model Evaluation for 4-Tier Medical Admissions Classification

This script provides detailed evaluation metrics including ordinal-aware metrics,
per-tier performance, and critical decision analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_recall_curve, 
    auc, log_loss, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.base import clone
from scipy.stats import kendalltau, ttest_rel
import joblib
import os
from datetime import datetime

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from four_tier_classifier import HighConfidenceFourTierClassifier

class OrdinalAwareMetrics:
    """
    Metrics that consider the ordinal nature of tiers
    """
    
    def __init__(self, cost_matrix=None):
        if cost_matrix is None:
            # Default: Penalize large jumps more than adjacent errors
            self.cost_matrix = np.array([
                [0, 1, 3, 5],   # True: Tier 1 (Very Unlikely)
                [2, 0, 1, 3],   # True: Tier 2 (Potential Review)
                [5, 3, 0, 1],   # True: Tier 3 (Probable Interview)
                [10, 5, 2, 0]   # True: Tier 4 (Very Likely Interview)
            ])
        else:
            self.cost_matrix = cost_matrix
    
    def weighted_f1_score(self, y_true, y_pred):
        """
        F1 that weights classes by importance and cost
        """
        # Standard F1 for each class
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Weight by class importance (Tier 4 most important)
        importance_weights = np.array([0.2, 0.25, 0.25, 0.3])
        
        # Additional penalty for extreme misclassifications
        cm = confusion_matrix(y_true, y_pred)
        total_cost = np.sum(cm * self.cost_matrix)
        cost_penalty = 1 / (1 + total_cost / len(y_true))
        
        weighted_f1 = np.sum(f1_scores * importance_weights) * cost_penalty
        return weighted_f1

def calculate_per_tier_metrics(y_true, y_pred, n_classes=4):
    """
    Calculate sensitivity, specificity, PPV, NPV for each tier
    """
    metrics = {}
    
    for tier in range(n_classes):
        # Convert to binary problem
        y_true_binary = (y_true == tier).astype(int)
        y_pred_binary = (y_pred == tier).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases where one class is missing
            if cm.shape == (1, 1):
                if y_true_binary.sum() == 0:  # All negative
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:  # All positive
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics[f'tier_{tier+1}'] = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,          # Precision
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
    
    return metrics

def ordinal_accuracy_score(y_true, y_pred, tolerance=0):
    """
    Accuracy allowing errors within 'tolerance' tiers
    tolerance=0: exact match required
    tolerance=1: adjacent tiers considered correct
    """
    diff = np.abs(y_true - y_pred)
    return np.mean(diff <= tolerance)

def mean_absolute_error_tiers(y_true, y_pred):
    """
    Average distance between predicted and true tier
    """
    return np.mean(np.abs(y_true - y_pred))

def kendall_tau_score(y_true, y_pred):
    """
    Correlation between predicted and true rankings
    Perfect ordering = 1.0
    """
    try:
        return kendalltau(y_true, y_pred)[0]
    except:
        return 0.0

def interview_decision_metrics(y_true, y_pred):
    """
    Binary metrics for the most important decision:
    Interview (Tiers 3-4) vs No Interview (Tiers 1-2)
    """
    y_true_interview = (y_true >= 2).astype(int)
    y_pred_interview = (y_pred >= 2).astype(int)
    
    cm = confusion_matrix(y_true_interview, y_pred_interview)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, cm.sum()
    
    return {
        'interview_sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Don't miss good candidates
        'interview_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Don't overwhelm with interviews
        'interview_ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,         # Interview yield
        'missed_good_candidates': fn,             # Critical error count
        'unnecessary_interviews': fp              # Resource waste count
    }

def multiclass_pr_auc(y_true, y_proba, n_classes=4):
    """
    Calculate AUPRC for each class in one-vs-rest fashion
    """
    try:
        # Binarize the labels
        y_true_binary = label_binarize(y_true, classes=range(n_classes))
        
        pr_auc_scores = {}
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_binary[:, i], 
                y_proba[:, i]
            )
            pr_auc_scores[f'tier_{i+1}_auprc'] = auc(recall, precision)
        
        # Weighted average by class frequency
        class_weights = np.bincount(y_true, minlength=n_classes) / len(y_true)
        weighted_avg = sum(pr_auc_scores[f'tier_{i+1}_auprc'] * class_weights[i] 
                          for i in range(n_classes))
        
        pr_auc_scores['weighted_avg_auprc'] = weighted_avg
        
        return pr_auc_scores
    except Exception as e:
        print(f"Warning: Could not calculate AUPRC: {e}")
        return {f'tier_{i+1}_auprc': 0.0 for i in range(n_classes)}

class ComprehensiveEvaluator:
    """
    Complete evaluation suite for 4-tier classification
    """
    
    def __init__(self, tier_names=['Very Unlikely', 'Potential Review', 
                                   'Probable Interview', 'Very Likely Interview']):
        self.tier_names = tier_names
        self.metrics_history = []
        self.ordinal_metrics = OrdinalAwareMetrics()
        
    def evaluate(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """
        Calculate all relevant metrics
        """
        results = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'n_samples': len(y_true)
        }
        
        # 1. Overall metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # 2. F1 variants
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_custom_weighted'] = self.ordinal_metrics.weighted_f1_score(y_true, y_pred)
        
        # 3. Ordinal metrics
        results['ordinal_accuracy_exact'] = ordinal_accuracy_score(y_true, y_pred, 0)
        results['ordinal_accuracy_adjacent'] = ordinal_accuracy_score(y_true, y_pred, 1)
        results['mae_tiers'] = mean_absolute_error_tiers(y_true, y_pred)
        results['kendall_tau'] = kendall_tau_score(y_true, y_pred)
        
        # 4. Cost metrics
        cm = confusion_matrix(y_true, y_pred)
        total_cost = np.sum(cm * self.ordinal_metrics.cost_matrix[:cm.shape[0], :cm.shape[1]])
        results['total_cost'] = total_cost
        results['avg_cost_per_sample'] = total_cost / len(y_true)
        
        # 5. Per-tier metrics
        tier_metrics = calculate_per_tier_metrics(y_true, y_pred)
        for tier, metrics in tier_metrics.items():
            for metric_name, value in metrics.items():
                results[f'{tier}_{metric_name}'] = value
        
        # 6. Critical decision metrics
        interview_metrics = interview_decision_metrics(y_true, y_pred)
        results.update(interview_metrics)
        
        # 7. Probabilistic metrics (if probabilities provided)
        if y_proba is not None:
            pr_auc = multiclass_pr_auc(y_true, y_proba)
            results.update(pr_auc)
            
            # Log loss
            try:
                results['log_loss'] = log_loss(y_true, y_proba)
            except:
                results['log_loss'] = float('inf')
        
        self.metrics_history.append(results)
        return results
    
    def create_detailed_report(self, results):
        """
        Generate comprehensive human-readable report
        """
        report = f"""
# Evaluation Metrics for 4-Class Medical Admissions Model

## Model: {results['model_name']}
**Evaluation Date:** {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
**Sample Size:** {results['n_samples']} applicants

---

## 🎯 Primary Performance Metrics

### Overall Classification Performance
- **Accuracy:** {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)
- **Balanced Accuracy:** {results['balanced_accuracy']:.3f} (better for imbalanced classes)
- **Cohen's Kappa:** {results['cohen_kappa']:.3f} (agreement beyond chance)

### F1 Score Variants
- **F1 (Micro):** {results['f1_micro']:.3f}
- **F1 (Macro):** {results['f1_macro']:.3f} 
- **F1 (Weighted):** {results['f1_weighted']:.3f}
- **F1 (Custom Weighted):** {results['f1_custom_weighted']:.3f} ⭐ *Ordinal-aware with cost penalties*

---

## 📊 Ordinal Classification Performance

### Tier Accuracy Analysis
- **Exact Tier Match:** {results['ordinal_accuracy_exact']:.3f} ({results['ordinal_accuracy_exact']*100:.1f}%)
- **Within 1 Tier:** {results['ordinal_accuracy_adjacent']:.3f} ({results['ordinal_accuracy_adjacent']*100:.1f}%) ⭐ *Key metric*
- **Average Tier Error:** {results['mae_tiers']:.2f} tiers
- **Ranking Correlation (Kendall's τ):** {results['kendall_tau']:.3f}

### Cost Analysis
- **Total Misclassification Cost:** {results['total_cost']:.0f}
- **Average Cost per Sample:** {results['avg_cost_per_sample']:.2f} ⭐ *Lower is better*

---

## 🎯 Critical Decision Analysis

### Interview Decision Performance
*Binary classification: Interview (Tiers 3-4) vs No Interview (Tiers 1-2)*

- **Interview Sensitivity:** {results['interview_sensitivity']:.3f} ({results['interview_sensitivity']*100:.1f}%) ⭐ *Don't miss good candidates*
- **Interview Specificity:** {results['interview_specificity']:.3f} ({results['interview_specificity']*100:.1f}%) ⭐ *Avoid interview overload*
- **Interview PPV:** {results['interview_ppv']:.3f} ({results['interview_ppv']*100:.1f}%) *Interview yield*

### Critical Error Counts
- **Missed Good Candidates:** {results['missed_good_candidates']} ❌ *Should be minimized*
- **Unnecessary Interviews:** {results['unnecessary_interviews']} ⚠️ *Resource waste*

---

## 📋 Per-Tier Detailed Performance

"""
        
        for tier in range(4):
            tier_name = self.tier_names[tier]
            report += f"""
### Tier {tier+1}: {tier_name}
- **Sensitivity (Recall):** {results[f'tier_{tier+1}_sensitivity']:.3f} *How well we identify this tier*
- **Specificity:** {results[f'tier_{tier+1}_specificity']:.3f} *How well we avoid false positives*
- **PPV (Precision):** {results[f'tier_{tier+1}_ppv']:.3f} *When we predict this tier, how often correct*
- **NPV:** {results[f'tier_{tier+1}_npv']:.3f} *When we don't predict this tier, how often correct*
- **F1 Score:** {results[f'tier_{tier+1}_f1']:.3f} *Balanced precision/recall*
"""

        # Add probabilistic metrics if available
        if 'log_loss' in results and results['log_loss'] != float('inf'):
            report += f"""
---

## 📈 Probabilistic Performance

### Probability Calibration
- **Log Loss:** {results['log_loss']:.3f} *Lower indicates better calibrated probabilities*

### Area Under Precision-Recall Curve (AUPRC)
"""
            for tier in range(4):
                if f'tier_{tier+1}_auprc' in results:
                    report += f"- **Tier {tier+1} AUPRC:** {results[f'tier_{tier+1}_auprc']:.3f}\n"
            
            if 'weighted_avg_auprc' in results:
                report += f"- **Weighted Average AUPRC:** {results['weighted_avg_auprc']:.3f} ⭐\n"

        # Performance interpretation
        report += f"""
---

## 🚦 Performance Interpretation

### 🟢 Strengths
"""
        
        # Identify strengths
        if results['interview_sensitivity'] >= 0.85:
            report += "- ✅ **Excellent interview sensitivity** - Rarely missing good candidates\n"
        if results['ordinal_accuracy_adjacent'] >= 0.80:
            report += "- ✅ **Strong ordinal performance** - Most predictions within 1 tier\n"
        if results['avg_cost_per_sample'] <= 1.0:
            report += "- ✅ **Low misclassification cost** - Efficient decision making\n"
        
        report += "\n### 🟡 Areas for Improvement\n"
        
        # Identify weaknesses
        if results['interview_sensitivity'] < 0.80:
            report += "- ⚠️ **Low interview sensitivity** - May be missing good candidates\n"
        if results['interview_specificity'] < 0.70:
            report += "- ⚠️ **Low interview specificity** - May be scheduling too many interviews\n"
        if results['ordinal_accuracy_adjacent'] < 0.75:
            report += "- ⚠️ **Poor ordinal accuracy** - Many predictions off by 2+ tiers\n"
        if results['avg_cost_per_sample'] > 1.5:
            report += "- ⚠️ **High misclassification cost** - Expensive prediction errors\n"

        report += f"""
---

## 🎯 Recommended Focus Areas

### Primary Metrics to Monitor
1. **Interview Sensitivity** ≥ 0.85 (currently {results['interview_sensitivity']:.3f})
2. **Ordinal Accuracy (±1 tier)** ≥ 0.80 (currently {results['ordinal_accuracy_adjacent']:.3f})
3. **Average Cost per Sample** ≤ 1.0 (currently {results['avg_cost_per_sample']:.2f})

### Secondary Metrics
1. **F1 Custom Weighted** (currently {results['f1_custom_weighted']:.3f})
2. **Tier 4 PPV** ≥ 0.75 (currently {results['tier_4_ppv']:.3f})
3. **Missed Good Candidates** ≤ 10% of total (currently {results['missed_good_candidates']})

---
*Report generated by Comprehensive Medical Admissions Evaluator*
"""
        
        return report

def load_and_combine_data():
    """Load and combine 2022 and 2023 data."""
    DATA_2022_PATH = "/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    DATA_2023_PATH = "/Users/JCR/Downloads/AdmissionsDataset/2023 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    
    print("Loading 2022 data...")
    pipeline_2022 = HighConfidenceFourTierClassifier(DATA_2022_PATH)
    df_2022 = pipeline_2022._prepare_data_for_training()
    df_2022['data_year'] = 2022
    
    print("Loading 2023 data...")
    pipeline_2023 = HighConfidenceFourTierClassifier(DATA_2023_PATH)
    df_2023 = pipeline_2023._prepare_data_for_training()
    df_2023['data_year'] = 2023
    
    # Combine datasets
    df_combined = pd.concat([df_2022, df_2023], ignore_index=True)
    
    # Create target mapping
    class_map = {label: idx for idx, label in enumerate(pipeline_2022.class_names)}
    df_combined["tier_int"] = df_combined["target"].map(class_map)
    
    print(f"Combined dataset shape: {df_combined.shape}")
    print(f"Class distribution:\n{df_combined['target'].value_counts()}")
    
    return df_combined, pipeline_2022.features, class_map, pipeline_2022.class_names

def evaluate_model_comprehensive_cv(name, model, X, y, evaluator, cv_folds=10):
    """Evaluate model using comprehensive cross-validation."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"\nEvaluating {name} with {cv_folds}-fold comprehensive CV...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = model_clone.predict(X_val)
        try:
            y_proba = model_clone.predict_proba(X_val)
        except:
            y_proba = None
        
        # Comprehensive evaluation
        fold_result = evaluator.evaluate(y_val, y_pred, y_proba, f"{name}_fold_{fold+1}")
        fold_results.append(fold_result)
    
    # Aggregate results
    aggregated = aggregate_cv_results(fold_results, name)
    return aggregated, fold_results

def aggregate_cv_results(fold_results, model_name):
    """Aggregate cross-validation results across folds."""
    # Get all numeric metrics
    numeric_metrics = {}
    for key in fold_results[0].keys():
        if isinstance(fold_results[0][key], (int, float)) and key not in ['timestamp', 'n_samples']:
            values = [result[key] for result in fold_results if not np.isnan(result[key])]
            if values:
                numeric_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # Create summary
    summary = {
        'model_name': model_name,
        'cv_folds': len(fold_results),
        'total_samples': sum(result['n_samples'] for result in fold_results),
        'metrics': numeric_metrics
    }
    
    return summary

def main():
    print("Starting comprehensive model evaluation with combined 2022-2023 data...")
    
    # Load data
    df_combined, features, class_map, class_names = load_and_combine_data()
    
    # Prepare features and target
    X = df_combined[features].values
    y = df_combined["tier_int"].values
    
    # Scale features
    features_to_scale_idx = []
    for i, feat in enumerate(features):
        if any(keyword in feat for keyword in ['Hour', 'Hours', 'Trend', 'Age', 'Num_Dependents']) or \
           feat in ['research_intensity', 'clinical_intensity', 'experience_balance', 'service_commitment', 'adversity_overcome']:
            if feat not in ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']:
                features_to_scale_idx.append(i)
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    if features_to_scale_idx:
        X_scaled[:, features_to_scale_idx] = scaler.fit_transform(X[:, features_to_scale_idx])
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(tier_names=class_names)
    
    # Define models
    models = {
        "RandomForestBaseline": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(150, 100, 50), 
            max_iter=2000, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.01
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1
        )
    
    # Evaluate all models
    all_results = {}
    all_fold_results = {}
    
    for name, model in models.items():
        aggregated, fold_results = evaluate_model_comprehensive_cv(
            name, model, X_scaled, y, evaluator, cv_folds=10
        )
        all_results[name] = aggregated
        all_fold_results[name] = fold_results
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION: {name}")
        print(f"{'='*80}")
        
        metrics = aggregated['metrics']
        print(f"Accuracy: {metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f}")
        print(f"F1 (Weighted): {metrics['f1_weighted']['mean']:.3f} ± {metrics['f1_weighted']['std']:.3f}")
        print(f"F1 (Custom): {metrics['f1_custom_weighted']['mean']:.3f} ± {metrics['f1_custom_weighted']['std']:.3f}")
        print(f"Ordinal Accuracy (±1): {metrics['ordinal_accuracy_adjacent']['mean']:.3f} ± {metrics['ordinal_accuracy_adjacent']['std']:.3f}")
        print(f"Interview Sensitivity: {metrics['interview_sensitivity']['mean']:.3f} ± {metrics['interview_sensitivity']['std']:.3f}")
        print(f"Avg Cost per Sample: {metrics['avg_cost_per_sample']['mean']:.3f} ± {metrics['avg_cost_per_sample']['std']:.3f}")
    
    # Generate detailed reports for each model
    print(f"\n{'='*100}")
    print("DETAILED MODEL REPORTS")
    print(f"{'='*100}")
    
    for name, model in models.items():
        # Train on full dataset for detailed analysis
        model.fit(X_scaled, y)
        y_pred_full = model.predict(X_scaled)
        try:
            y_proba_full = model.predict_proba(X_scaled)
        except:
            y_proba_full = None
        
        # Generate comprehensive evaluation
        full_results = evaluator.evaluate(y, y_pred_full, y_proba_full, name)
        
        # Create and save detailed report
        detailed_report = evaluator.create_detailed_report(full_results)
        
        # Save report to file
        report_filename = f"detailed_evaluation_report_{name.lower()}.md"
        with open(report_filename, 'w') as f:
            f.write(detailed_report)
        
        print(f"\nDetailed report for {name} saved to: {report_filename}")
        
        # Print key metrics
        print(f"\n{name} Key Metrics (Full Dataset):")
        print(f"  Interview Sensitivity: {full_results['interview_sensitivity']:.3f}")
        print(f"  Ordinal Accuracy (±1): {full_results['ordinal_accuracy_adjacent']:.3f}")
        print(f"  Custom Weighted F1: {full_results['f1_custom_weighted']:.3f}")
        print(f"  Avg Cost per Sample: {full_results['avg_cost_per_sample']:.3f}")
    
    # Save all results
    results_file = 'comprehensive_evaluation_results.pkl'
    joblib.dump({
        'aggregated_results': all_results,
        'fold_results': all_fold_results,
        'evaluation_config': {
            'features': features,
            'class_names': class_names,
            'total_samples': len(X),
            'cv_folds': 10
        }
    }, results_file)
    
    print(f"\nAll comprehensive results saved to: {results_file}")
    
    # Final recommendation
    best_model = max(all_results.keys(), 
                    key=lambda x: all_results[x]['metrics']['f1_custom_weighted']['mean'])
    
    print(f"\n{'='*100}")
    print(f"FINAL RECOMMENDATION: {best_model}")
    print(f"{'='*100}")
    print("Based on custom weighted F1 score (ordinal-aware with cost penalties)")
    print(f"Best CV Performance: {all_results[best_model]['metrics']['f1_custom_weighted']['mean']:.3f}")

if __name__ == "__main__":
    main()
