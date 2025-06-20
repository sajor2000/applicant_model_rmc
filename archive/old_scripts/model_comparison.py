"""Model comparison script to evaluate alternative approaches on 4-tier admissions problem.

This script combines 2022 and 2023 data for comprehensive training and uses 10-fold cross-validation
to evaluate models. The best model is trained on 100% of the combined dataset for optimal performance.

Models evaluated:
1. Baseline RandomForestClassifier (current approach)
2. OrdinalClassifier – threshold-based ordinal regression that respects the natural tier order
3. CostSensitiveRF – RandomForest with sample-weights derived from an asymmetric cost matrix
4. XGBoostClassifier – Gradient boosting with advanced regularization
5. NeuralNetworkClassifier – Multi-layer perceptron with dropout

The code re-uses the data-preparation pipeline of HighConfidenceFourTierClassifier to guarantee
identical feature engineering & preprocessing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    make_scorer
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from scipy import stats
import joblib

# Try to import XGBoost, fall back gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Re-use existing pipeline for data loading / preprocessing
from four_tier_classifier import HighConfidenceFourTierClassifier

################################################################################
# Helper models
################################################################################

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Simple threshold-based ordinal regression using repeated binary classifiers.

    Train *K-1* binary classifiers that learn P(y > k | x).  Adapted from Frank & Hall (2001).
    """

    def __init__(self, base_estimator: BaseEstimator | None = None):
        self.base_estimator = base_estimator or LogisticRegression(max_iter=1000)
        self.classifiers_: list[BaseEstimator] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray):
        self.classes_ = np.sort(np.unique(y))
        self.classifiers_.clear()
        for threshold in self.classes_[:-1]:
            binary_y = (y > threshold).astype(int)
            clf = clone(self.base_estimator)
            # Handle both DataFrame and numpy array inputs
            X_input = X.values if hasattr(X, 'values') else X
            clf.fit(X_input, binary_y)
            self.classifiers_.append(clf)
        return self

    def _cum_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return cumulative P(y > k) for each threshold k."""
        cum_probs = np.zeros((X.shape[0], len(self.classes_) - 1))
        for idx, clf in enumerate(self.classifiers_):
            # Handle both DataFrame and numpy array inputs
            X_input = X.values if hasattr(X, 'values') else X
            cum_probs[:, idx] = clf.predict_proba(X_input)[:, 1]
        return cum_probs

    def predict_proba(self, X: pd.DataFrame | np.ndarray):
        cum = self._cum_proba(X)
        n_classes = len(self.classes_)
        probs = np.zeros((X.shape[0], n_classes))
        probs[:, 0] = 1 - cum[:, 0]
        for k in range(1, n_classes - 1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, -1] = cum[:, -1]
        return probs

    def predict(self, X: pd.DataFrame | np.ndarray):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

################################################################################
# Cost-sensitive RandomForest (Fixed)
################################################################################

def create_cost_matrix() -> np.ndarray:
    """Custom misclassification cost matrix (rows:true, cols:pred).
    
    Higher costs for more severe misclassifications, especially
    rejecting strong candidates (Very Likely -> Very Unlikely).
    """
    return np.array(
        [
            [0, 1, 3, 5],   # Very Unlikely
            [2, 0, 1, 3],   # Potential Review
            [5, 3, 0, 1],   # Probable Interview
            [10, 5, 2, 0],  # Very Likely Interview
        ]
    )

class CostSensitiveRF(RandomForestClassifier):
    """RandomForest that uses sample-weights derived from an asymmetric cost matrix.
    
    Fixed implementation: weights samples by the total cost of misclassifying their class.
    """

    def __init__(self, cost_matrix: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.cost_matrix = cost_matrix

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        
        # Weight each instance by total cost of misclassifying that class
        # (sum of off-diagonal elements for that row)
        for idx, true_class in enumerate(y):
            true_class_int = int(true_class)
            # Total cost of misclassifying this class
            misclass_cost = (self.cost_matrix[true_class_int].sum() - 
                           self.cost_matrix[true_class_int, true_class_int])
            sample_weight[idx] *= max(misclass_cost, 1.0)  # Ensure minimum weight of 1
            
        # Handle both DataFrame and numpy array inputs
        X_input = X.values if hasattr(X, 'values') else X
        return super().fit(X_input, y, sample_weight=sample_weight)

################################################################################
# Wrapper for consistent scaling
################################################################################

class ScaledClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that applies consistent scaling to any classifier."""
    
    def __init__(self, base_classifier: BaseEstimator, features_to_scale: List[str] = None):
        self.base_classifier = base_classifier
        self.features_to_scale = features_to_scale or []
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.scale_cols_ = []
        
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_scaled = X.copy()
            if self.features_to_scale:
                self.scale_cols_ = [col for col in self.features_to_scale if col in X.columns]
                if self.scale_cols_:
                    X_scaled[self.scale_cols_] = self.scaler.fit_transform(X[self.scale_cols_])
            # Pass the scaled DataFrame to the base classifier
            return self.base_classifier.fit(X_scaled, y, **fit_params)
        else:
            # If numpy array, scale all features
            X_scaled = self.scaler.fit_transform(X)
            return self.base_classifier.fit(X_scaled, y, **fit_params)
    
    def predict(self, X: pd.DataFrame | np.ndarray):
        X_scaled = self._transform_X(X)
        return self.base_classifier.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray):
        X_scaled = self._transform_X(X)
        return self.base_classifier.predict_proba(X_scaled)
    
    def _transform_X(self, X: pd.DataFrame | np.ndarray):
        if isinstance(X, pd.DataFrame):
            X_scaled = X.copy()
            if self.scale_cols_:
                X_scaled[self.scale_cols_] = self.scaler.transform(X[self.scale_cols_])
            return X_scaled
        else:
            # If numpy array, scale all features
            return self.scaler.transform(X)

################################################################################
# Evaluation utilities
################################################################################

def calculate_total_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Calculate average cost per prediction using the cost matrix."""
    total = 0.0
    for t, p in zip(y_true, y_pred):
        total += cost_matrix[int(t), int(p)]
    return total / len(y_true)

def cost_scorer(cost_matrix: np.ndarray):
    """Create a scorer function for cross-validation that uses the cost matrix."""
    def _cost_score(y_true, y_pred):
        return -calculate_total_cost(y_true, y_pred, cost_matrix)  # Negative because sklearn maximizes
    return make_scorer(_cost_score)

def statistical_significance_test(scores1: np.ndarray, scores2: np.ndarray, 
                                model1_name: str, model2_name: str) -> Dict:
    """Perform paired t-test to check if difference in scores is statistically significant."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    mean_diff = np.mean(scores1) - np.mean(scores2)
    
    result = {
        'model1': model1_name,
        'model2': model2_name,
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better_model': model1_name if mean_diff > 0 else model2_name
    }
    return result

def evaluate_model_cv_only(name: str, model: BaseEstimator, 
                          X: pd.DataFrame, y: np.ndarray, 
                          cost_matrix: np.ndarray, cv_folds: int = 10) -> Dict:
    """Comprehensive model evaluation using only cross-validation."""
    
    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    try:
        cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        cv_cost = cross_val_score(model, X, y, cv=cv, scoring=cost_scorer(cost_matrix))
        cv_cost = -cv_cost  # Convert back to positive (was negated for maximization)
        
        # Check for NaN values and handle them
        if np.isnan(cv_accuracy).any():
            print(f"Warning: NaN values detected in accuracy scores for {name}")
            cv_accuracy = cv_accuracy[~np.isnan(cv_accuracy)]
        if np.isnan(cv_f1).any():
            print(f"Warning: NaN values detected in F1 scores for {name}")
            cv_f1 = cv_f1[~np.isnan(cv_f1)]
        if np.isnan(cv_cost).any():
            print(f"Warning: NaN values detected in cost scores for {name}")
            cv_cost = cv_cost[~np.isnan(cv_cost)]
            
    except Exception as e:
        print(f"Error in cross-validation for {name}: {e}")
        # Fallback to manual cross-validation
        cv_accuracy, cv_f1, cv_cost = manual_cross_validation(model, X, y, cost_matrix, cv)
    
    results = {
        'model_name': name,
        'cv_accuracy_mean': np.mean(cv_accuracy) if len(cv_accuracy) > 0 else 0.0,
        'cv_accuracy_std': np.std(cv_accuracy) if len(cv_accuracy) > 0 else 0.0,
        'cv_f1_mean': np.mean(cv_f1) if len(cv_f1) > 0 else 0.0,
        'cv_f1_std': np.std(cv_f1) if len(cv_f1) > 0 else 0.0,
        'cv_cost_mean': np.mean(cv_cost) if len(cv_cost) > 0 else 0.0,
        'cv_cost_std': np.std(cv_cost) if len(cv_cost) > 0 else 0.0,
        'cv_accuracy_scores': cv_accuracy,
        'cv_f1_scores': cv_f1,
        'cv_cost_scores': cv_cost
    }
    
    return results

def manual_cross_validation(model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, 
                           cost_matrix: np.ndarray, cv) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manual cross-validation as fallback."""
    print("Performing manual cross-validation...")
    
    cv_accuracy = []
    cv_f1 = []
    cv_cost = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
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
            
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            continue
    
    return np.array(cv_accuracy), np.array(cv_f1), np.array(cv_cost)

def print_cv_results(results: Dict):
    """Print cross-validation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Model: {results['model_name']}")
    print(f"{'='*60}")
    
    print(f"\n--- 10-Fold Cross-Validation Performance ---")
    print(f"Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print(f"Weighted F1: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    print(f"Cost per sample: {results['cv_cost_mean']:.4f} ± {results['cv_cost_std']:.4f}")

def get_feature_importance(model: BaseEstimator, feature_names: List[str]) -> Dict:
    """Extract feature importance from model if available."""
    importance_dict = {}
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        # Linear models - use absolute values of coefficients
        if len(model.coef_.shape) == 1:
            importances = np.abs(model.coef_)
        else:
            # Multi-class: average absolute coefficients across classes
            importances = np.mean(np.abs(model.coef_), axis=0)
        importance_dict = dict(zip(feature_names, importances))
    elif hasattr(model, 'base_classifier') and hasattr(model.base_classifier, 'feature_importances_'):
        # Wrapped models
        importances = model.base_classifier.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
    
    return importance_dict

################################################################################
# Data loading utilities
################################################################################

def load_and_combine_data(data_2022_path: str, data_2023_path: str) -> Tuple[pd.DataFrame, list[str], Dict[str, int]]:
    """Load and combine 2022 and 2023 data using the existing pipeline."""
    print("Loading 2022 data...")
    pipeline_2022 = HighConfidenceFourTierClassifier(data_2022_path)
    df_2022 = pipeline_2022._prepare_data_for_training()
    df_2022['data_year'] = 2022
    
    print("Loading 2023 data...")
    pipeline_2023 = HighConfidenceFourTierClassifier(data_2023_path)
    df_2023 = pipeline_2023._prepare_data_for_training()
    df_2023['data_year'] = 2023
    
    # Ensure both datasets have the same features
    common_features = list(set(pipeline_2022.features) & set(pipeline_2023.features))
    print(f"Common features between datasets: {len(common_features)}")
    
    # Use features from 2022 as baseline (should be the same)
    features = pipeline_2022.features
    class_map = {label: idx for idx, label in enumerate(pipeline_2022.class_names)}
    
    # Combine datasets
    df_combined = pd.concat([df_2022, df_2023], ignore_index=True)
    df_combined["tier_int"] = df_combined["target"].map(class_map)
    
    print(f"Combined dataset shape: {df_combined.shape}")
    print(f"Class distribution:\n{df_combined['target'].value_counts()}")
    
    return df_combined, features, class_map

################################################################################
# Main routine
################################################################################

def main():
    print("Starting comprehensive model comparison with combined 2022-2023 data...")
    
    # Data paths
    DATA_2022_PATH = Path("/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
    DATA_2023_PATH = Path("/Users/JCR/Downloads/AdmissionsDataset/2023 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
    
    if not DATA_2022_PATH.exists():
        raise FileNotFoundError(f"Could not locate 2022 data at {DATA_2022_PATH}")
    if not DATA_2023_PATH.exists():
        raise FileNotFoundError(f"Could not locate 2023 data at {DATA_2023_PATH}")

    # Load and combine data
    df_combined, features, class_map = load_and_combine_data(str(DATA_2022_PATH), str(DATA_2023_PATH))
    X = df_combined[features]  # Keep as DataFrame for consistent scaling
    y = df_combined["tier_int"].values  # 0-3 integers

    # Identify features to scale (same logic as original classifier)
    features_to_scale = [col for col in features if 'Hour' in col or 'Hours' in col or 'Trend' in col or 
                        col in ['Age', 'Num_Dependents', 'research_intensity', 'clinical_intensity', 
                               'experience_balance', 'service_commitment', 'adversity_overcome']]
    features_to_scale = [col for col in features_to_scale if col not in ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']]
    
    print(f"Features to be scaled: {features_to_scale}")

    cost_matrix = create_cost_matrix()
    print(f"Using cost matrix:\n{cost_matrix}")

    # Define models with consistent scaling
    models: Dict[str, BaseEstimator] = {
        "RandomForestBaseline": ScaledClassifier(
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
            features_to_scale
        ),
        "OrdinalClassifier": ScaledClassifier(
            OrdinalClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")),
            features_to_scale
        ),
        "CostSensitiveRF": ScaledClassifier(
            CostSensitiveRF(cost_matrix, n_estimators=200, random_state=42, n_jobs=-1),
            features_to_scale
        ),
        "NeuralNetwork": ScaledClassifier(
            MLPClassifier(
                hidden_layer_sizes=(150, 100, 50), 
                max_iter=2000, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01,  # L2 regularization
                learning_rate='adaptive'
            ),
            features_to_scale
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = ScaledClassifier(
            xgb.XGBClassifier(
                n_estimators=200,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0,
                n_jobs=-1
            ),
            features_to_scale
        )

    # Evaluate all models using 10-fold cross-validation
    print(f"\nEvaluating {len(models)} models using 10-fold cross-validation...")
    all_results = {}
    
    for name, model in models.items():
        print(f"\nProcessing {name}...")
        results = evaluate_model_cv_only(
            name, model, X, y, cost_matrix, cv_folds=10
        )
        all_results[name] = results
        print_cv_results(results)

    # Statistical significance testing
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    
    model_names = list(all_results.keys())
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            # Test on CV accuracy scores
            sig_test = statistical_significance_test(
                all_results[model1]['cv_accuracy_scores'],
                all_results[model2]['cv_accuracy_scores'],
                model1, model2
            )
            
            print(f"\n{model1} vs {model2}:")
            print(f"  Mean accuracy difference: {sig_test['mean_diff']:.4f}")
            print(f"  P-value: {sig_test['p_value']:.4f}")
            print(f"  Statistically significant: {sig_test['significant']}")
            if sig_test['significant']:
                print(f"  Better model: {sig_test['better_model']}")

    # Summary comparison table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame({
        'Model': [results['model_name'] for results in all_results.values()],
        'CV_Accuracy_Mean': [results['cv_accuracy_mean'] for results in all_results.values()],
        'CV_Accuracy_Std': [results['cv_accuracy_std'] for results in all_results.values()],
        'CV_F1_Mean': [results['cv_f1_mean'] for results in all_results.values()],
        'CV_F1_Std': [results['cv_f1_std'] for results in all_results.values()],
        'CV_Cost_Mean': [results['cv_cost_mean'] for results in all_results.values()],
        'CV_Cost_Std': [results['cv_cost_std'] for results in all_results.values()]
    })
    
    # Sort by CV accuracy (most reliable metric)
    summary_df = summary_df.sort_values('CV_Accuracy_Mean', ascending=False)
    print(summary_df.round(4).to_string(index=False))
    
    # Train the best model on 100% of the data
    best_model_name = summary_df.iloc[0]['Model']
    print(f"\n{'='*80}")
    print(f"TRAINING BEST MODEL ON 100% OF DATA: {best_model_name}")
    print(f"{'='*80}")
    
    best_model = models[best_model_name]
    print(f"Training {best_model_name} on full dataset ({len(X)} samples)...")
    best_model.fit(X, y)
    
    # Feature importance analysis for best model
    importance = get_feature_importance(best_model, features)
    if importance:
        print(f"\n--- Feature Importances for {best_model_name} ---")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features:
            print(f"{feat}: {imp:.4f}")
    
    # Save the best model
    model_save_path = f'models/best_model_{best_model_name.lower()}_combined_2022_2023.pkl'
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': best_model,
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
    results_file = 'model_comparison_results_combined_2022_2023.pkl'
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
