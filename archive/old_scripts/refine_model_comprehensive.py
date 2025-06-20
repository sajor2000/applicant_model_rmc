"""
Comprehensive Model Refinement with Advanced Techniques
=======================================================

Implements extensive hyperparameter search, ensemble methods, calibration,
and other advanced techniques to maximize model performance and confidence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, GridSearchCV,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (
    make_scorer, cohen_kappa_score, accuracy_score,
    precision_recall_curve, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# Custom scorer for adjacent accuracy
def adjacent_accuracy_scorer(y_true, y_pred):
    """Score based on predictions being within 1 bucket."""
    return np.mean(np.abs(y_true - y_pred) <= 1)

adjacent_scorer = make_scorer(adjacent_accuracy_scorer, greater_is_better=True)


class AdvancedCascadeOptimizer:
    """Advanced optimization for cascade classifier."""
    
    def __init__(self):
        self.best_params = {}
        self.best_models = {}
        self.calibrators = {}
        
    def load_training_data(self):
        """Load and prepare training data with all features."""
        print("Loading training data...")
        
        # Load filtered data
        df_2022 = pd.read_excel("data_filtered/2022_filtered_applicants.xlsx")
        df_2023 = pd.read_excel("data_filtered/2023_filtered_applicants.xlsx")
        
        # Load LLM scores
        llm_train = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
        llm_train = llm_train.rename(columns={'AMCAS_ID_original': 'amcas_id'})
        
        # Merge
        llm_cols = [col for col in llm_train.columns if col.startswith('llm_')]
        df_2022 = df_2022.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
        df_2023 = df_2023.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
        
        df_train = pd.concat([df_2022, df_2023], ignore_index=True)
        
        return df_train
    
    def engineer_advanced_features(self, df):
        """Create advanced features for better discrimination."""
        print("Engineering advanced features...")
        
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Profile coherence features
        if 'llm_overall_essay_score' in df.columns:
            # Essay-structure alignment
            struct_features = ['healthcare_total_hours', 'exp_hour_research', 
                             'exp_hour_volunteer_med', 'service_rating_numerical']
            essay_features = ['llm_clinical_insight', 'llm_service_genuineness',
                            'llm_motivation_authenticity', 'llm_leadership_impact']
            
            # Normalize features for correlation
            for feat in struct_features + essay_features:
                if feat in df.columns:
                    df[f'{feat}_norm'] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-6)
            
            # Coherence score
            df['profile_coherence'] = 0
            for s_feat in struct_features:
                for e_feat in essay_features:
                    if f'{s_feat}_norm' in df.columns and f'{e_feat}_norm' in df.columns:
                        df['profile_coherence'] += df[f'{s_feat}_norm'] * df[f'{e_feat}_norm']
            
            # Essay variance (indicates mixed signals)
            essay_cols = [col for col in df.columns if col.startswith('llm_') and 
                         col.endswith(('score', 'authenticity', 'depth', 'impact'))]
            if len(essay_cols) > 1:
                df['essay_variance'] = df[essay_cols].var(axis=1)
                df['essay_consistency'] = 1 / (1 + df['essay_variance'])
        
        # Experience diversity
        exp_cols = ['exp_hour_research', 'exp_hour_clinical', 'exp_hour_volunteer_med',
                   'exp_hour_volunteer_non_med', 'exp_hour_shadowing']
        available_exp = [col for col in exp_cols if col in df.columns]
        if len(available_exp) > 1:
            df['experience_diversity'] = (df[available_exp] > 0).sum(axis=1)
            df['experience_balance'] = 1 - df[available_exp].std(axis=1) / (df[available_exp].mean(axis=1) + 1)
        
        # Extreme indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['amcas_id', 'application_review_score']:
                p95 = df[col].quantile(0.95)
                p5 = df[col].quantile(0.05)
                df[f'{col}_extreme_high'] = (df[col] > p95).astype(int)
                df[f'{col}_extreme_low'] = (df[col] < p5).astype(int)
        
        # Interaction features for key predictors
        if 'service_rating_numerical' in df.columns:
            # Service × Essay quality
            if 'llm_overall_essay_score' in df.columns:
                df['service_essay_interaction'] = (
                    df['service_rating_numerical'] * df['llm_overall_essay_score'] / 100
                )
            
            # Service × Clinical experience
            if 'healthcare_total_hours' in df.columns:
                df['service_clinical_interaction'] = (
                    df['service_rating_numerical'] * 
                    np.log1p(df['healthcare_total_hours'])
                )
        
        return df
    
    def prepare_cascade_data(self, df):
        """Prepare data for cascade stages."""
        # Get features and target
        exclude_cols = ['application_review_score', 'amcas_id', 'appl_year', 
                       'year', 'AMCAS_ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['application_review_score'].values
        
        # Handle categorical features
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1},
            'citizenship': {'US_Citizen': 0, 'Permanent_Resident': 1, 
                          'International': 2, 'Other': 3},
            'service_rating_categorical': {
                'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                'Good': 2, 'Average': 1, 'Below Average': 0, 'Poor': -1
            }
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(-2)
        
        # Convert remaining categoricals
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        return X_scaled, y, feature_cols, imputer, scaler
    
    def optimize_stage_hyperparameters(self, X, y, stage_name):
        """Extensive hyperparameter optimization for a stage."""
        print(f"\nOptimizing {stage_name}...")
        
        # Define extensive parameter grid
        param_distributions = {
            'n_estimators': randint(400, 1000),
            'max_depth': randint(3, 8),
            'learning_rate': uniform(0.005, 0.1),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'colsample_bylevel': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0.5, 3),
            'gamma': uniform(0, 1),
            'scale_pos_weight': uniform(0.5, 2)
        }
        
        # Use Optuna for Bayesian optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.5),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.5),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            model = XGBClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=-1)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        
        print(f"Best AUC for {stage_name}: {study.best_value:.4f}")
        
        return best_params
    
    def train_ensemble_stage(self, X, y, stage_params, stage_name):
        """Train ensemble of models for a stage."""
        print(f"\nTraining ensemble for {stage_name}...")
        
        models = []
        
        # 1. XGBoost with optimized params
        xgb_model = XGBClassifier(**stage_params)
        
        # 2. LightGBM
        lgb_params = {
            'n_estimators': stage_params['n_estimators'],
            'learning_rate': stage_params['learning_rate'],
            'max_depth': stage_params['max_depth'],
            'num_leaves': 2 ** stage_params['max_depth'] - 1,
            'subsample': stage_params['subsample'],
            'colsample_bytree': stage_params['colsample_bytree'],
            'reg_alpha': stage_params['reg_alpha'],
            'reg_lambda': stage_params['reg_lambda'],
            'random_state': 42,
            'verbosity': -1
        }
        lgb_model = LGBMClassifier(**lgb_params)
        
        # 3. CatBoost
        cat_params = {
            'iterations': stage_params['n_estimators'],
            'learning_rate': stage_params['learning_rate'],
            'depth': stage_params['max_depth'],
            'l2_leaf_reg': stage_params['reg_lambda'],
            'random_state': 42,
            'verbose': False
        }
        cat_model = CatBoostClassifier(**cat_params)
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('cat', cat_model)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Calibrate probabilities
        calibrated = CalibratedClassifierCV(
            ensemble, method='isotonic', cv=3
        )
        calibrated.fit(X, y)
        
        return calibrated
    
    def train_optimized_cascade(self, df_train):
        """Train fully optimized cascade classifier."""
        print("\n" + "="*80)
        print("TRAINING OPTIMIZED CASCADE CLASSIFIER")
        print("="*80)
        
        # Engineer advanced features
        df_enhanced = self.engineer_advanced_features(df_train)
        
        # Prepare data
        X, y, feature_cols, imputer, scaler = self.prepare_cascade_data(df_enhanced)
        
        # Stage 1: Reject vs Non-Reject
        print("\n### STAGE 1: Reject vs Non-Reject ###")
        y_stage1 = (y > 9).astype(int)
        
        # Optimize hyperparameters
        stage1_params = self.optimize_stage_hyperparameters(X, y_stage1, "Stage 1")
        self.best_params['stage1'] = stage1_params
        
        # Train ensemble
        stage1_model = self.train_ensemble_stage(X, y_stage1, stage1_params, "Stage 1")
        self.best_models['stage1'] = stage1_model
        
        # Stage 2: Waitlist vs Higher
        print("\n### STAGE 2: Waitlist vs Higher ###")
        mask_stage2 = y > 9
        X_stage2 = X[mask_stage2]
        y_stage2 = (y[mask_stage2] > 15).astype(int)
        
        stage2_params = self.optimize_stage_hyperparameters(X_stage2, y_stage2, "Stage 2")
        self.best_params['stage2'] = stage2_params
        
        stage2_model = self.train_ensemble_stage(X_stage2, y_stage2, stage2_params, "Stage 2")
        self.best_models['stage2'] = stage2_model
        
        # Stage 3: Interview vs Accept
        print("\n### STAGE 3: Interview vs Accept ###")
        mask_stage3 = y > 15
        X_stage3 = X[mask_stage3]
        y_stage3 = (y[mask_stage3] >= 23).astype(int)
        
        stage3_params = self.optimize_stage_hyperparameters(X_stage3, y_stage3, "Stage 3")
        self.best_params['stage3'] = stage3_params
        
        stage3_model = self.train_ensemble_stage(X_stage3, y_stage3, stage3_params, "Stage 3")
        self.best_models['stage3'] = stage3_model
        
        return feature_cols, imputer, scaler
    
    def predict_with_confidence(self, X):
        """Make predictions with enhanced confidence scoring."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        probabilities = np.zeros((n_samples, 4))
        confidences = np.zeros(n_samples)
        
        # Stage 1
        prob_not_reject = self.best_models['stage1'].predict_proba(X)[:, 1]
        stage1_confidence = np.maximum(prob_not_reject, 1 - prob_not_reject)
        
        # Reject decisions
        is_reject = prob_not_reject < 0.5
        predictions[is_reject] = 0
        probabilities[is_reject, 0] = 1 - prob_not_reject[is_reject]
        confidences[is_reject] = stage1_confidence[is_reject]
        
        # Non-reject: proceed to stage 2
        non_reject_mask = ~is_reject
        if non_reject_mask.sum() > 0:
            X_stage2 = X[non_reject_mask]
            
            # Stage 2
            prob_higher = self.best_models['stage2'].predict_proba(X_stage2)[:, 1]
            stage2_confidence = np.maximum(prob_higher, 1 - prob_higher)
            
            # Waitlist decisions
            is_waitlist = prob_higher < 0.5
            waitlist_indices = np.where(non_reject_mask)[0][is_waitlist]
            predictions[waitlist_indices] = 1
            probabilities[waitlist_indices, 1] = 1 - prob_higher[is_waitlist]
            
            # Combine confidences
            non_reject_conf = confidences[non_reject_mask]
            non_reject_conf[is_waitlist] = (
                stage1_confidence[non_reject_mask][is_waitlist] * 
                stage2_confidence[is_waitlist]
            ) ** 0.5
            confidences[non_reject_mask] = non_reject_conf
            
            # Higher: proceed to stage 3
            higher_mask = np.zeros(n_samples, dtype=bool)
            higher_indices = np.where(non_reject_mask)[0][~is_waitlist]
            higher_mask[higher_indices] = True
            
            if higher_mask.sum() > 0:
                X_stage3 = X[higher_mask]
                
                # Stage 3
                prob_accept = self.best_models['stage3'].predict_proba(X_stage3)[:, 1]
                stage3_confidence = np.maximum(prob_accept, 1 - prob_accept)
                
                # Accept/Interview decisions
                is_accept = prob_accept >= 0.5
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                predictions[accept_indices] = 3
                predictions[interview_indices] = 2
                
                probabilities[accept_indices, 3] = prob_accept[is_accept]
                probabilities[interview_indices, 2] = 1 - prob_accept[~is_accept]
                
                # Final confidence
                higher_conf = confidences[higher_mask]
                higher_conf = (
                    stage1_confidence[higher_mask] * 
                    stage2_confidence[~is_waitlist] * 
                    stage3_confidence
                ) ** (1/3)
                confidences[higher_mask] = higher_conf
        
        # Convert confidence to 0-100 scale
        confidences = confidences * 100
        
        return predictions, probabilities, confidences
    
    def evaluate_on_test(self, df_test, feature_cols, imputer, scaler):
        """Evaluate optimized model on test data."""
        print("\n" + "="*80)
        print("EVALUATING ON 2024 TEST DATA")
        print("="*80)
        
        # Engineer same features
        df_test_enhanced = self.engineer_advanced_features(df_test)
        
        # Prepare test data
        X_test, y_test, _, _, _ = self.prepare_cascade_data(df_test_enhanced)
        
        # Predict
        predictions, probabilities, confidences = self.predict_with_confidence(X_test)
        
        # Convert to buckets
        true_buckets = np.zeros(len(y_test), dtype=int)
        true_buckets[y_test <= 9] = 0
        true_buckets[(y_test >= 10) & (y_test <= 15)] = 1
        true_buckets[(y_test >= 16) & (y_test <= 22)] = 2
        true_buckets[y_test >= 23] = 3
        
        # Calculate metrics
        exact_match = np.mean(predictions == true_buckets)
        adjacent = np.mean(np.abs(predictions - true_buckets) <= 1)
        
        # Confidence distribution
        low_conf = (confidences < 60).sum()
        med_conf = ((confidences >= 60) & (confidences < 80)).sum()
        high_conf = (confidences >= 80).sum()
        
        print(f"\nPerformance Metrics:")
        print(f"Exact Match: {exact_match:.3f}")
        print(f"Adjacent Accuracy: {adjacent:.3f}")
        
        print(f"\nConfidence Distribution:")
        print(f"Low (<60): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
        print(f"Medium (60-80): {med_conf} ({med_conf/len(confidences)*100:.1f}%)")
        print(f"High (>80): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
        
        return predictions, confidences, exact_match, adjacent


def main():
    # Initialize optimizer
    optimizer = AdvancedCascadeOptimizer()
    
    # Load training data
    df_train = optimizer.load_training_data()
    
    # Train optimized cascade
    feature_cols, imputer, scaler = optimizer.train_optimized_cascade(df_train)
    
    # Load test data
    df_test = pd.read_excel("data_filtered/2024_filtered_applicants.xlsx")
    llm_test_files = list(Path(".").glob("llm_scores_2024_*.csv"))
    if llm_test_files:
        llm_test = pd.read_csv(sorted(llm_test_files)[-1])
        llm_test = llm_test.rename(columns={'AMCAS_ID_original': 'amcas_id'})
        llm_cols = [col for col in llm_test.columns if col.startswith('llm_')]
        df_test = df_test.merge(llm_test[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    
    # Evaluate
    predictions, confidences, exact_match, adjacent = optimizer.evaluate_on_test(
        df_test, feature_cols, imputer, scaler
    )
    
    # Save optimized model
    print("\nSaving optimized model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_data = {
        'optimizer': optimizer,
        'feature_cols': feature_cols,
        'imputer': imputer,
        'scaler': scaler,
        'performance': {
            'exact_match': exact_match,
            'adjacent': adjacent
        },
        'hyperparameters': optimizer.best_params
    }
    
    model_path = f"models/optimized_cascade_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()