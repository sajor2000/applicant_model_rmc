"""
Comprehensive Model Refinement with Extended Hyperparameter Search
==================================================================

This script performs extensive hyperparameter optimization over 30-40 minutes
to maximize model performance and reduce low confidence predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, GridSearchCV,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    make_scorer, cohen_kappa_score, accuracy_score,
    precision_recall_curve, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import optuna
from scipy.stats import uniform, randint, loguniform
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import time
import warnings
warnings.filterwarnings('ignore')

# Get CPU count and use all available cores
n_cpus = multiprocessing.cpu_count()
n_jobs = n_cpus - 1  # Leave one core for system
print(f"Using {n_jobs} out of {n_cpus} CPU cores")

# Custom scorer for adjacent accuracy
def adjacent_accuracy_scorer(y_true, y_pred):
    """Score based on predictions being within 1 bucket."""
    return np.mean(np.abs(y_true - y_pred) <= 1)

adjacent_scorer = make_scorer(adjacent_accuracy_scorer, greater_is_better=True)


class ComprehensiveCascadeOptimizer:
    """Comprehensive optimization with extended search."""
    
    def __init__(self):
        self.best_params = {}
        self.best_models = {}
        self.feature_importance = {}
        self.optimization_history = {}
        self.start_time = time.time()
        self.n_jobs = n_jobs
        
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
        print(f"Total training samples: {len(df_train)}")
        
        return df_train
    
    def engineer_comprehensive_features(self, df):
        """Create comprehensive feature set."""
        print("Engineering comprehensive features...")
        
        df = df.copy()
        
        # 1. Profile Coherence Features
        if 'llm_overall_essay_score' in df.columns:
            # Essay-structure alignment
            struct_features = ['healthcare_total_hours', 'exp_hour_research', 
                             'exp_hour_volunteer_med', 'service_rating_numerical']
            essay_features = ['llm_clinical_insight', 'llm_service_genuineness',
                            'llm_motivation_authenticity', 'llm_leadership_impact']
            
            # Normalize features
            for feat in struct_features + essay_features:
                if feat in df.columns:
                    df[f'{feat}_norm'] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-6)
            
            # Profile coherence score
            df['profile_coherence'] = 0
            coherence_count = 0
            for s_feat in struct_features:
                for e_feat in essay_features:
                    if f'{s_feat}_norm' in df.columns and f'{e_feat}_norm' in df.columns:
                        df['profile_coherence'] += df[f'{s_feat}_norm'] * df[f'{e_feat}_norm']
                        coherence_count += 1
            
            if coherence_count > 0:
                df['profile_coherence'] = df['profile_coherence'] / coherence_count
            
            # Essay variance
            essay_cols = [col for col in df.columns if col.startswith('llm_') and 
                         col.endswith(('score', 'authenticity', 'depth', 'impact'))]
            if len(essay_cols) > 1:
                df['essay_variance'] = df[essay_cols].var(axis=1)
                df['essay_consistency'] = 1 / (1 + df['essay_variance'])
                df['essay_mean'] = df[essay_cols].mean(axis=1)
                df['essay_skewness'] = df[essay_cols].skew(axis=1)
        
        # 2. Experience Features
        exp_cols = ['exp_hour_research', 'exp_hour_clinical', 'exp_hour_volunteer_med',
                   'exp_hour_volunteer_non_med', 'exp_hour_shadowing', 'healthcare_total_hours']
        available_exp = [col for col in exp_cols if col in df.columns]
        
        if len(available_exp) > 1:
            # Experience diversity
            df['experience_diversity'] = (df[available_exp] > 0).sum(axis=1)
            df['experience_balance'] = 1 - df[available_exp].std(axis=1) / (df[available_exp].mean(axis=1) + 1)
            
            # Log-transformed experience
            for col in available_exp:
                df[f'{col}_log'] = np.log1p(df[col])
            
            # Experience ratios
            if 'healthcare_total_hours' in df.columns and 'exp_hour_research' in df.columns:
                df['clinical_research_ratio'] = df['healthcare_total_hours'] / (df['exp_hour_research'] + 1)
            
            if 'exp_hour_volunteer_med' in df.columns and 'exp_hour_volunteer_non_med' in df.columns:
                df['med_nonmed_volunteer_ratio'] = df['exp_hour_volunteer_med'] / (df['exp_hour_volunteer_non_med'] + 1)
        
        # 3. Service Rating Features
        if 'service_rating_numerical' in df.columns:
            df['service_squared'] = df['service_rating_numerical'] ** 2
            df['service_cubed'] = df['service_rating_numerical'] ** 3
            df['service_sqrt'] = np.sqrt(df['service_rating_numerical'])
            
            # Service interactions
            if 'llm_overall_essay_score' in df.columns:
                df['service_essay_product'] = df['service_rating_numerical'] * df['llm_overall_essay_score'] / 25
                df['service_essay_ratio'] = df['service_rating_numerical'] / (df['llm_overall_essay_score'] / 25 + 1)
            
            if 'healthcare_total_hours' in df.columns:
                df['service_clinical_product'] = df['service_rating_numerical'] * np.log1p(df['healthcare_total_hours'])
                df['service_clinical_ratio'] = df['service_rating_numerical'] / (np.log1p(df['healthcare_total_hours']) + 1)
            
            if 'age' in df.columns:
                df['service_age_product'] = df['service_rating_numerical'] * df['age']
        
        # 4. Flag Features
        if 'llm_red_flag_count' in df.columns and 'llm_green_flag_count' in df.columns:
            df['flag_balance'] = df['llm_green_flag_count'] - df['llm_red_flag_count']
            df['flag_ratio'] = df['llm_green_flag_count'] / (df['llm_red_flag_count'] + 1)
            df['total_flags'] = df['llm_green_flag_count'] + df['llm_red_flag_count']
            df['flag_net_score'] = (df['llm_green_flag_count'] - df['llm_red_flag_count']) / (df['total_flags'] + 1)
        
        # 5. Demographic Interactions
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['age_log'] = np.log(df['age'])
            
            if 'first_generation_ind' in df.columns:
                df['age_first_gen'] = df['age'] * df['first_generation_ind']
            
            if 'pell_grant_ind' in df.columns:
                df['age_pell'] = df['age'] * df['pell_grant_ind']
        
        # 6. Extreme Value Indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['extreme_value_count'] = 0
        
        for col in numeric_cols:
            if col not in ['amcas_id', 'application_review_score', 'extreme_value_count']:
                p95 = df[col].quantile(0.95)
                p5 = df[col].quantile(0.05)
                df[f'{col}_extreme_high'] = (df[col] > p95).astype(int)
                df[f'{col}_extreme_low'] = (df[col] < p5).astype(int)
                df['extreme_value_count'] += ((df[col] > p95) | (df[col] < p5)).astype(int)
        
        # 7. Composite Scores
        # Academic potential score
        academic_features = ['exp_hour_research', 'llm_intellectual_curiosity', 'llm_maturity_score']
        available_academic = [f for f in academic_features if f in df.columns]
        if len(available_academic) > 0:
            for feat in available_academic:
                df[f'{feat}_academic_norm'] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min() + 1e-6)
            
            academic_norm_cols = [f'{feat}_academic_norm' for feat in available_academic]
            df['academic_potential_score'] = df[academic_norm_cols].mean(axis=1)
        
        # Clinical readiness score
        clinical_features = ['healthcare_total_hours', 'llm_clinical_insight', 'exp_hour_shadowing']
        available_clinical = [f for f in clinical_features if f in df.columns]
        if len(available_clinical) > 0:
            for feat in available_clinical:
                df[f'{feat}_clinical_norm'] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min() + 1e-6)
            
            clinical_norm_cols = [f'{feat}_clinical_norm' for feat in available_clinical]
            df['clinical_readiness_score'] = df[clinical_norm_cols].mean(axis=1)
        
        # Clean up temporary normalized columns
        temp_cols = [col for col in df.columns if '_norm' in col or '_academic_norm' in col or '_clinical_norm' in col]
        df = df.drop(columns=temp_cols)
        
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
    
    def extensive_hyperparameter_search(self, X, y, stage_name):
        """Extensive hyperparameter search using Optuna with time budget."""
        print(f"\n{'='*60}")
        print(f"EXTENSIVE OPTIMIZATION FOR {stage_name}")
        print(f"{'='*60}")
        
        elapsed_time = (time.time() - self.start_time) / 60
        remaining_time = max(5, 35 - elapsed_time)  # Allocate time per stage
        time_per_stage = remaining_time / (4 - len(self.best_params))  # Remaining stages
        
        print(f"Time elapsed: {elapsed_time:.1f} minutes")
        print(f"Time allocated for this stage: {time_per_stage:.1f} minutes")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.2, 5),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'n_jobs': self.n_jobs
            }
            
            # Try different boosters
            booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])
            params['booster'] = booster
            
            if booster == 'dart':
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 0, 0.3)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0, 0.5)
            
            model = XGBClassifier(**params)
            
            # Use stratified k-fold cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='roc_auc', n_jobs=self.n_jobs
            )
            
            # Also calculate adjacent accuracy
            adj_scores = cross_val_score(
                model, X, y, cv=5, scoring=adjacent_scorer, n_jobs=self.n_jobs
            )
            
            # Combined metric: prioritize AUC but consider adjacent accuracy
            return 0.7 * cv_scores.mean() + 0.3 * adj_scores.mean()
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Optimize with time limit
        study.optimize(
            objective, 
            n_trials=None,
            timeout=time_per_stage * 60,  # Convert to seconds
            n_jobs=1  # Optuna handles parallelization internally
        )
        
        print(f"\nCompleted {len(study.trials)} trials")
        print(f"Best score: {study.best_value:.4f}")
        
        # Get best parameters
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        best_params['tree_method'] = 'hist'
        best_params['n_jobs'] = self.n_jobs
        
        # Save optimization history
        self.optimization_history[stage_name] = {
            'n_trials': len(study.trials),
            'best_score': study.best_value,
            'best_params': best_params,
            'time_spent': time_per_stage
        }
        
        return best_params
    
    def train_ensemble_stage(self, X, y, stage_params, stage_name):
        """Train ensemble of XGBoost models with different configurations."""
        print(f"\nTraining ensemble for {stage_name}...")
        
        models = []
        
        # Base model with best params
        base_model = XGBClassifier(**stage_params)
        base_model.fit(X, y)
        models.append(('base', base_model))
        
        # Variant 1: More trees, lower learning rate
        variant1_params = stage_params.copy()
        variant1_params['n_estimators'] = int(variant1_params['n_estimators'] * 1.5)
        variant1_params['learning_rate'] = variant1_params['learning_rate'] * 0.7
        variant1_params['random_state'] = 123
        variant1 = XGBClassifier(**variant1_params)
        variant1.fit(X, y)
        models.append(('variant1', variant1))
        
        # Variant 2: Deeper trees, more regularization
        variant2_params = stage_params.copy()
        variant2_params['max_depth'] = min(variant2_params['max_depth'] + 2, 12)
        variant2_params['reg_alpha'] = variant2_params.get('reg_alpha', 0) * 1.5
        variant2_params['reg_lambda'] = variant2_params.get('reg_lambda', 1) * 1.5
        variant2_params['random_state'] = 456
        variant2 = XGBClassifier(**variant2_params)
        variant2.fit(X, y)
        models.append(('variant2', variant2))
        
        # Variant 3: Different subsample strategy
        variant3_params = stage_params.copy()
        variant3_params['subsample'] = max(variant3_params.get('subsample', 0.8) * 0.9, 0.5)
        variant3_params['colsample_bytree'] = max(variant3_params.get('colsample_bytree', 0.8) * 0.9, 0.5)
        variant3_params['random_state'] = 789
        variant3 = XGBClassifier(**variant3_params)
        variant3.fit(X, y)
        models.append(('variant3', variant3))
        
        # Variant 4: Conservative model
        variant4_params = stage_params.copy()
        variant4_params['n_estimators'] = int(variant4_params['n_estimators'] * 0.8)
        variant4_params['max_depth'] = max(variant4_params['max_depth'] - 1, 3)
        variant4_params['min_child_weight'] = variant4_params.get('min_child_weight', 1) * 2
        variant4_params['random_state'] = 321
        variant4 = XGBClassifier(**variant4_params)
        variant4.fit(X, y)
        models.append(('variant4', variant4))
        
        # Create voting ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=self.n_jobs)
        ensemble.fit(X, y)
        
        # Calibrate probabilities using isotonic regression
        calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        calibrated.fit(X, y)
        
        # Store feature importance from base model
        self.feature_importance[stage_name] = base_model.feature_importances_
        
        print(f"  Ensemble created with {len(models)} models")
        
        return calibrated
    
    def train_optimized_cascade(self, df_train):
        """Train fully optimized cascade classifier."""
        print("\n" + "="*80)
        print("TRAINING COMPREHENSIVE CASCADE CLASSIFIER")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Engineer comprehensive features
        df_enhanced = self.engineer_comprehensive_features(df_train)
        
        # Prepare data
        X, y, feature_cols, imputer, scaler = self.prepare_cascade_data(df_enhanced)
        
        # Save feature columns
        self.feature_cols = feature_cols
        
        # Stage 1: Reject vs Non-Reject
        print("\n### STAGE 1: Reject vs Non-Reject ###")
        y_stage1 = (y > 9).astype(int)
        print(f"Class distribution: Reject={np.sum(y_stage1==0)}, Non-Reject={np.sum(y_stage1==1)}")
        
        # Extensive optimization
        stage1_params = self.extensive_hyperparameter_search(X, y_stage1, "Stage 1")
        self.best_params['stage1'] = stage1_params
        
        # Train ensemble
        stage1_model = self.train_ensemble_stage(X, y_stage1, stage1_params, "Stage 1")
        self.best_models['stage1'] = stage1_model
        
        # Cross-validation performance
        cv_scores = cross_val_score(stage1_model, X, y_stage1, cv=5, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 1 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 2: Waitlist vs Higher
        print("\n### STAGE 2: Waitlist vs Higher ###")
        mask_stage2 = y > 9
        X_stage2 = X[mask_stage2]
        y_stage2 = (y[mask_stage2] > 15).astype(int)
        print(f"Class distribution: Waitlist={np.sum(y_stage2==0)}, Higher={np.sum(y_stage2==1)}")
        
        stage2_params = self.extensive_hyperparameter_search(X_stage2, y_stage2, "Stage 2")
        self.best_params['stage2'] = stage2_params
        
        stage2_model = self.train_ensemble_stage(X_stage2, y_stage2, stage2_params, "Stage 2")
        self.best_models['stage2'] = stage2_model
        
        cv_scores = cross_val_score(stage2_model, X_stage2, y_stage2, cv=5, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 2 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 3: Interview vs Accept
        print("\n### STAGE 3: Interview vs Accept ###")
        mask_stage3 = y > 15
        X_stage3 = X[mask_stage3]
        y_stage3 = (y[mask_stage3] >= 23).astype(int)
        print(f"Class distribution: Interview={np.sum(y_stage3==0)}, Accept={np.sum(y_stage3==1)}")
        
        stage3_params = self.extensive_hyperparameter_search(X_stage3, y_stage3, "Stage 3")
        self.best_params['stage3'] = stage3_params
        
        stage3_model = self.train_ensemble_stage(X_stage3, y_stage3, stage3_params, "Stage 3")
        self.best_models['stage3'] = stage3_model
        
        cv_scores = cross_val_score(stage3_model, X_stage3, y_stage3, cv=5, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 3 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Training complete
        total_time = (time.time() - self.start_time) / 60
        print(f"\nTotal training time: {total_time:.1f} minutes")
        
        return feature_cols, imputer, scaler
    
    def predict_with_enhanced_confidence(self, X):
        """Make predictions with enhanced confidence scoring."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        probabilities = np.zeros((n_samples, 4))
        confidences = np.zeros(n_samples)
        
        # Stage 1 predictions
        stage1_proba = self.best_models['stage1'].predict_proba(X)
        prob_not_reject = stage1_proba[:, 1]
        
        # Enhanced confidence using margin and entropy
        prob_margin = np.abs(prob_not_reject - 0.5) * 2  # 0 to 1
        entropy = -prob_not_reject * np.log(prob_not_reject + 1e-10) - (1 - prob_not_reject) * np.log(1 - prob_not_reject + 1e-10)
        entropy_norm = 1 - entropy / np.log(2)  # 0 to 1, higher is more confident
        
        stage1_confidence = (prob_margin + entropy_norm) / 2
        
        # Reject decisions
        is_reject = prob_not_reject < 0.5
        predictions[is_reject] = 0
        probabilities[is_reject, 0] = 1 - prob_not_reject[is_reject]
        confidences[is_reject] = stage1_confidence[is_reject]
        
        # Non-reject: proceed to stage 2
        non_reject_mask = ~is_reject
        if non_reject_mask.sum() > 0:
            X_stage2 = X[non_reject_mask]
            
            # Stage 2 predictions
            stage2_proba = self.best_models['stage2'].predict_proba(X_stage2)
            prob_higher = stage2_proba[:, 1]
            
            # Stage 2 confidence
            prob_margin2 = np.abs(prob_higher - 0.5) * 2
            entropy2 = -prob_higher * np.log(prob_higher + 1e-10) - (1 - prob_higher) * np.log(1 - prob_higher + 1e-10)
            entropy_norm2 = 1 - entropy2 / np.log(2)
            stage2_confidence = (prob_margin2 + entropy_norm2) / 2
            
            # Waitlist decisions
            is_waitlist = prob_higher < 0.5
            waitlist_indices = np.where(non_reject_mask)[0][is_waitlist]
            predictions[waitlist_indices] = 1
            probabilities[waitlist_indices, 1] = 1 - prob_higher[is_waitlist]
            
            # Combine confidences using harmonic mean (more conservative)
            non_reject_conf = confidences[non_reject_mask]
            stage1_conf_subset = stage1_confidence[non_reject_mask]
            non_reject_conf[is_waitlist] = 2 / (1/stage1_conf_subset[is_waitlist] + 1/stage2_confidence[is_waitlist])
            confidences[non_reject_mask] = non_reject_conf
            
            # Higher: proceed to stage 3
            higher_mask = np.zeros(n_samples, dtype=bool)
            higher_indices = np.where(non_reject_mask)[0][~is_waitlist]
            higher_mask[higher_indices] = True
            
            if higher_mask.sum() > 0:
                X_stage3 = X[higher_mask]
                
                # Stage 3 predictions
                stage3_proba = self.best_models['stage3'].predict_proba(X_stage3)
                prob_accept = stage3_proba[:, 1]
                
                # Stage 3 confidence
                prob_margin3 = np.abs(prob_accept - 0.5) * 2
                entropy3 = -prob_accept * np.log(prob_accept + 1e-10) - (1 - prob_accept) * np.log(1 - prob_accept + 1e-10)
                entropy_norm3 = 1 - entropy3 / np.log(2)
                stage3_confidence = (prob_margin3 + entropy_norm3) / 2
                
                # Accept/Interview decisions
                is_accept = prob_accept >= 0.5
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                predictions[accept_indices] = 3
                predictions[interview_indices] = 2
                
                probabilities[accept_indices, 3] = prob_accept[is_accept]
                probabilities[interview_indices, 2] = 1 - prob_accept[~is_accept]
                
                # Final confidence (harmonic mean of all stages)
                higher_conf = confidences[higher_mask]
                stage1_conf_higher = stage1_confidence[higher_mask]
                stage2_conf_higher = stage2_confidence[~is_waitlist]
                
                # Three-way harmonic mean
                higher_conf = 3 / (1/stage1_conf_higher + 1/stage2_conf_higher + 1/stage3_confidence)
                confidences[higher_mask] = higher_conf
        
        # Fill in remaining probabilities
        for i in range(n_samples):
            prob_sum = probabilities[i].sum()
            if prob_sum < 0.99:
                probabilities[i, predictions[i]] = 1 - prob_sum
        
        # Convert confidence to 0-100 scale with calibration
        # Use sigmoid transformation for better distribution
        confidences = 100 / (1 + np.exp(-8 * (confidences - 0.5)))
        
        return predictions, probabilities, confidences
    
    def evaluate_on_test(self, df_test, feature_cols, imputer, scaler):
        """Evaluate optimized model on test data."""
        print("\n" + "="*80)
        print("EVALUATING ON 2024 TEST DATA")
        print("="*80)
        
        # Engineer same features
        df_test_enhanced = self.engineer_comprehensive_features(df_test)
        
        # Prepare test data
        X_test, y_test, _, _, _ = self.prepare_cascade_data(df_test_enhanced)
        
        # Predict
        predictions, probabilities, confidences = self.predict_with_enhanced_confidence(X_test)
        
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
        
        # Confusion matrix
        cm = confusion_matrix(true_buckets, predictions)
        bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
        
        print("\nConfusion Matrix:")
        print("True\\Pred   " + "  ".join(f"{name:>10}" for name in bucket_names))
        for i, true_name in enumerate(bucket_names):
            row = f"{true_name:10} "
            for j in range(4):
                row += f"{cm[i,j]:>10} "
            print(row)
        
        # Detailed metrics by quartile
        print("\nDetailed Metrics by Quartile:")
        for i, name in enumerate(bucket_names):
            mask = true_buckets == i
            if mask.sum() > 0:
                acc = (predictions[mask] == i).mean()
                avg_conf = confidences[mask].mean()
                print(f"{name}: Accuracy={acc:.3f}, Avg Confidence={avg_conf:.1f}")
        
        return predictions, confidences, probabilities, exact_match, adjacent
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get max features
        max_features = max(len(imp) for imp in self.feature_importance.values())
        
        # Combine importance across stages
        combined_importance = np.zeros(max_features)
        
        for stage, importance in self.feature_importance.items():
            if len(importance) < max_features:
                padded = np.pad(importance, (0, max_features - len(importance)), mode='constant')
            else:
                padded = importance[:max_features]
            combined_importance += padded
        
        # Normalize
        combined_importance = combined_importance / combined_importance.sum()
        
        # Create dataframe
        feature_cols_to_use = list(self.feature_cols[:min(len(self.feature_cols), len(combined_importance))])
        
        importance_df = pd.DataFrame({
            'feature': feature_cols_to_use,
            'importance': combined_importance[:len(feature_cols_to_use)]
        }).sort_values('importance', ascending=False)
        
        print("\nTop 30 Most Important Features:")
        for idx, row in importance_df.head(30).iterrows():
            print(f"{row['feature']:50} {row['importance']:.4f}")
        
        # Visualize
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(30)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 30 Feature Importance (Combined Across Stages)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('comprehensive_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to comprehensive_feature_importance.png")
        
        return importance_df
    
    def save_optimization_report(self):
        """Save detailed optimization report."""
        report_path = f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total optimization time: {(time.time() - self.start_time) / 60:.1f} minutes\n\n")
            
            for stage, history in self.optimization_history.items():
                f.write(f"\n{stage} Optimization:\n")
                f.write(f"  Trials completed: {history['n_trials']}\n")
                f.write(f"  Best score: {history['best_score']:.4f}\n")
                f.write(f"  Time spent: {history['time_spent']:.1f} minutes\n")
                f.write(f"  Best parameters:\n")
                for param, value in history['best_params'].items():
                    f.write(f"    {param}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\nOptimization report saved to: {report_path}")


def main():
    # Initialize optimizer
    optimizer = ComprehensiveCascadeOptimizer()
    
    # Load training data
    df_train = optimizer.load_training_data()
    
    # Train optimized cascade with extensive search
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
    predictions, confidences, probabilities, exact_match, adjacent = optimizer.evaluate_on_test(
        df_test, feature_cols, imputer, scaler
    )
    
    # Analyze feature importance
    importance_df = optimizer.analyze_feature_importance()
    
    # Save optimization report
    optimizer.save_optimization_report()
    
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
        'hyperparameters': optimizer.best_params,
        'feature_importance': importance_df,
        'optimization_history': optimizer.optimization_history
    }
    
    model_path = f"models/comprehensive_cascade_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'amcas_id': df_test['amcas_id'],
        'true_score': df_test['application_review_score'],
        'predicted_bucket': predictions,
        'confidence': confidences,
        'reject_prob': probabilities[:, 0],
        'waitlist_prob': probabilities[:, 1],
        'interview_prob': probabilities[:, 2],
        'accept_prob': probabilities[:, 3]
    })
    
    results_path = f"comprehensive_predictions_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE OPTIMIZATION COMPLETE!")
    print(f"Total time: {(time.time() - optimizer.start_time) / 60:.1f} minutes")
    print("="*80)


if __name__ == "__main__":
    main()