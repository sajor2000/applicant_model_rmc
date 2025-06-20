"""Simplified model comparison script for 4-tier admissions problem.

This script combines 2022 and 2023 data and evaluates models using 10-fold cross-validation.
Focuses on core functionality without complex wrappers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from scipy import stats
import joblib
import os

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from four_tier_classifier import HighConfidenceFourTierClassifier

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
    
    return df_combined, pipeline_2022.features, class_map

def create_cost_matrix():
    """Custom misclassification cost matrix."""
    return np.array([
        [0, 1, 3, 5],   # Very Unlikely
        [2, 0, 1, 3],   # Potential Review
        [5, 3, 0, 1],   # Probable Interview
        [10, 5, 2, 0],  # Very Likely Interview
    ])

def calculate_total_cost(y_true, y_pred, cost_matrix):
    """Calculate average cost per prediction."""
    total = 0.0
    for t, p in zip(y_true, y_pred):
        total += cost_matrix[int(t), int(p)]
    return total / len(y_true)

def evaluate_model_cv(name, model, X, y, cost_matrix, cv_folds=10):
    """Evaluate model using cross-validation."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_accuracy = []
    cv_f1 = []
    cv_cost = []
    
    print(f"Evaluating {name} with {cv_folds}-fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model_clone.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        cost = calculate_total_cost(y_val, y_pred, cost_matrix)
        
        cv_accuracy.append(acc)
        cv_f1.append(f1)
        cv_cost.append(cost)
    
    return {
        'model_name': name,
        'cv_accuracy_mean': np.mean(cv_accuracy),
        'cv_accuracy_std': np.std(cv_accuracy),
        'cv_f1_mean': np.mean(cv_f1),
        'cv_f1_std': np.std(cv_f1),
        'cv_cost_mean': np.mean(cv_cost),
        'cv_cost_std': np.std(cv_cost),
        'cv_accuracy_scores': np.array(cv_accuracy)
    }

def get_feature_importance(model, feature_names):
    """Extract feature importance if available."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    return {}

def main():
    print("Starting simplified model comparison with combined 2022-2023 data...")
    
    # Load data
    df_combined, features, class_map = load_and_combine_data()
    
    # Prepare features and target
    X = df_combined[features].values
    y = df_combined["tier_int"].values
    
    # Identify features to scale
    features_to_scale_idx = []
    for i, feat in enumerate(features):
        if any(keyword in feat for keyword in ['Hour', 'Hours', 'Trend', 'Age', 'Num_Dependents']) or \
           feat in ['research_intensity', 'clinical_intensity', 'experience_balance', 'service_commitment', 'adversity_overcome']:
            if feat not in ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']:
                features_to_scale_idx.append(i)
    
    print(f"Features to scale: {[features[i] for i in features_to_scale_idx]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = X.copy()
    if features_to_scale_idx:
        X_scaled[:, features_to_scale_idx] = scaler.fit_transform(X[:, features_to_scale_idx])
    
    cost_matrix = create_cost_matrix()
    print(f"Using cost matrix:\n{cost_matrix}")
    
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
    
    for name, model in models.items():
        results = evaluate_model_cv(name, model, X_scaled, y, cost_matrix, cv_folds=10)
        all_results[name] = results
        
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        print(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        print(f"CV Weighted F1: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
        print(f"CV Cost: {results['cv_cost_mean']:.4f} ± {results['cv_cost_std']:.4f}")
    
    # Statistical significance testing
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    
    model_names = list(all_results.keys())
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            scores1 = all_results[model1]['cv_accuracy_scores']
            scores2 = all_results[model2]['cv_accuracy_scores']
            
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            mean_diff = np.mean(scores1) - np.mean(scores2)
            
            print(f"\n{model1} vs {model2}:")
            print(f"  Mean accuracy difference: {mean_diff:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Statistically significant: {p_value < 0.05}")
            if p_value < 0.05:
                better_model = model1 if mean_diff > 0 else model2
                print(f"  Better model: {better_model}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame({
        'Model': [results['model_name'] for results in all_results.values()],
        'CV_Accuracy_Mean': [results['cv_accuracy_mean'] for results in all_results.values()],
        'CV_Accuracy_Std': [results['cv_accuracy_std'] for results in all_results.values()],
        'CV_F1_Mean': [results['cv_f1_mean'] for results in all_results.values()],
        'CV_Cost_Mean': [results['cv_cost_mean'] for results in all_results.values()]
    })
    
    summary_df = summary_df.sort_values('CV_Accuracy_Mean', ascending=False)
    print(summary_df.round(4).to_string(index=False))
    
    # Train best model on full dataset
    best_model_name = summary_df.iloc[0]['Model']
    print(f"\n{'='*80}")
    print(f"TRAINING BEST MODEL ON 100% OF DATA: {best_model_name}")
    print(f"{'='*80}")
    
    best_model = models[best_model_name]
    print(f"Training {best_model_name} on full dataset ({len(X)} samples)...")
    best_model.fit(X_scaled, y)
    
    # Feature importance
    importance = get_feature_importance(best_model, features)
    if importance:
        print(f"\n--- Feature Importances for {best_model_name} ---")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features:
            print(f"{feat}: {imp:.4f}")
    
    # Save best model
    model_save_path = f'models/best_model_{best_model_name.lower()}_combined_2022_2023.pkl'
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'features_to_scale_idx': features_to_scale_idx,
        'feature_names': features,
        'class_names': list(class_map.keys()),
        'model_name': best_model_name,
        'cv_results': all_results[best_model_name],
        'training_data_years': [2022, 2023],
        'training_samples': len(X)
    }
    
    joblib.dump(model_data, model_save_path)
    print(f"\nBest model saved to: {model_save_path}")
    
    # Save all results
    results_file = 'model_comparison_results_simple_combined_2022_2023.pkl'
    joblib.dump(all_results, results_file)
    print(f"All results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print(f"FINAL RECOMMENDATION: {best_model_name}")
    print(f"{'='*80}")
    print(f"Cross-validation accuracy: {all_results[best_model_name]['cv_accuracy_mean']:.4f} ± {all_results[best_model_name]['cv_accuracy_std']:.4f}")
    print(f"Cross-validation F1: {all_results[best_model_name]['cv_f1_mean']:.4f} ± {all_results[best_model_name]['cv_f1_std']:.4f}")
    print(f"Cross-validation cost: {all_results[best_model_name]['cv_cost_mean']:.4f} ± {all_results[best_model_name]['cv_cost_std']:.4f}")
    print(f"Trained on {len(X)} samples from 2022-2023 data")

if __name__ == "__main__":
    main()
