"""
Fast Model Refinement with High CPU Utilization
===============================================

Optimized version focusing on key hyperparameters with parallel processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    make_scorer, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# Get CPU count and use 75%
n_cpus = multiprocessing.cpu_count()
n_jobs = int(n_cpus * 0.75)
print(f"Using {n_jobs} out of {n_cpus} CPU cores (75%)")


class FastCascadeOptimizer:
    """Fast optimization focusing on most impactful improvements."""
    
    def __init__(self):
        self.best_params = {}
        self.best_models = {}
        self.feature_importance = {}
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
    
    def engineer_key_features(self, df):
        """Create only the most impactful features for speed."""
        print("Engineering key features...")
        
        df = df.copy()
        
        # 1. Profile coherence score (high impact)
        if 'llm_overall_essay_score' in df.columns and 'service_rating_numerical' in df.columns:
            # Essay-service alignment
            essay_norm = df['llm_overall_essay_score'] / 100
            service_norm = (df['service_rating_numerical'] - 1) / 3
            df['essay_service_alignment'] = 1 - abs(essay_norm - service_norm)
            
            # Service × Essay interaction
            df['service_essay_interaction'] = (
                df['service_rating_numerical'] * df['llm_overall_essay_score'] / 25
            )
        
        # 2. Experience consistency
        exp_cols = ['exp_hour_research', 'exp_hour_volunteer_med', 
                   'exp_hour_volunteer_non_med', 'exp_hour_shadowing']
        available_exp = [col for col in exp_cols if col in df.columns]
        
        if len(available_exp) > 1:
            exp_mean = df[available_exp].mean(axis=1)
            exp_std = df[available_exp].std(axis=1)
            df['experience_consistency'] = 1 / (1 + exp_std / (exp_mean + 1))
            df['experience_diversity'] = (df[available_exp] > 50).sum(axis=1)
        
        # 3. LLM consistency
        llm_score_cols = [col for col in df.columns if col.startswith('llm_') and 
                         col not in ['llm_red_flag_count', 'llm_green_flag_count']]
        if len(llm_score_cols) > 2:
            df['llm_consistency'] = 1 / (1 + df[llm_score_cols].std(axis=1))
        
        # 4. Red/Green flag features
        if 'llm_red_flag_count' in df.columns and 'llm_green_flag_count' in df.columns:
            df['flag_balance'] = df['llm_green_flag_count'] - df['llm_red_flag_count']
            df['flag_ratio'] = df['llm_green_flag_count'] / (df['llm_red_flag_count'] + 1)
        
        # 5. Key interaction features
        if 'service_rating_numerical' in df.columns and 'healthcare_total_hours' in df.columns:
            df['service_x_clinical'] = (
                df['service_rating_numerical'] * np.log1p(df['healthcare_total_hours'])
            )
        
        # 6. Polynomial features for top predictors
        if 'service_rating_numerical' in df.columns:
            df['service_squared'] = df['service_rating_numerical'] ** 2
        
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
    
    def optimize_stage_fast(self, X, y, stage_name):
        """Fast hyperparameter optimization focusing on key parameters."""
        print(f"\nOptimizing {stage_name} (fast mode)...")
        
        # Focused parameter grid - only most impactful parameters
        param_distributions = {
            'n_estimators': [400, 600, 800],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [1, 2, 3]
        }
        
        # Calculate positive class weight
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        
        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=pos_weight,
            tree_method='hist',  # Faster training
            n_jobs=self.n_jobs
        )
        
        # Use only 30 random combinations for speed
        random_search = RandomizedSearchCV(
            xgb,
            param_distributions=param_distributions,
            n_iter=30,
            cv=3,  # Reduced folds for speed
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        best_params = random_search.best_params_
        best_params['scale_pos_weight'] = pos_weight
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        best_params['n_jobs'] = self.n_jobs
        
        print(f"Best AUC for {stage_name}: {random_search.best_score_:.4f}")
        
        return best_params
    
    def train_ensemble_stage(self, X, y, stage_params, stage_name):
        """Train ensemble of 3 XGBoost models with different seeds."""
        print(f"\nTraining ensemble for {stage_name}...")
        
        models = []
        
        # Train 3 models with different random seeds
        for i in range(3):
            params = stage_params.copy()
            params['random_state'] = i * 42
            
            # Slightly vary parameters
            params['n_estimators'] = int(params['n_estimators'] * (1 + i * 0.1))
            
            model = XGBClassifier(**params)
            model.fit(X, y)
            models.append(model)
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[(f'xgb_{i}', model) for i, model in enumerate(models)],
            voting='soft',
            n_jobs=self.n_jobs
        )
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Calibrate probabilities
        calibrated = CalibratedClassifierCV(
            ensemble, method='isotonic', cv=3
        )
        calibrated.fit(X, y)
        
        # Store feature importance from first model
        self.feature_importance[stage_name] = models[0].feature_importances_
        
        return calibrated
    
    def train_optimized_cascade(self, df_train):
        """Train optimized cascade classifier."""
        print("\n" + "="*80)
        print("TRAINING FAST OPTIMIZED CASCADE CLASSIFIER")
        print("="*80)
        
        # Engineer key features
        df_enhanced = self.engineer_key_features(df_train)
        
        # Prepare data
        X, y, feature_cols, imputer, scaler = self.prepare_cascade_data(df_enhanced)
        
        # Save feature columns
        self.feature_cols = feature_cols
        
        # Stage 1: Reject vs Non-Reject
        print("\n### STAGE 1: Reject vs Non-Reject ###")
        y_stage1 = (y > 9).astype(int)
        
        stage1_params = self.optimize_stage_fast(X, y_stage1, "Stage 1")
        self.best_params['stage1'] = stage1_params
        
        stage1_model = self.train_ensemble_stage(X, y_stage1, stage1_params, "Stage 1")
        self.best_models['stage1'] = stage1_model
        
        # Cross-validation
        cv_scores = cross_val_score(stage1_model, X, y_stage1, cv=3, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 1 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 2: Waitlist vs Higher
        print("\n### STAGE 2: Waitlist vs Higher ###")
        mask_stage2 = y > 9
        X_stage2 = X[mask_stage2]
        y_stage2 = (y[mask_stage2] > 15).astype(int)
        
        stage2_params = self.optimize_stage_fast(X_stage2, y_stage2, "Stage 2")
        self.best_params['stage2'] = stage2_params
        
        stage2_model = self.train_ensemble_stage(X_stage2, y_stage2, stage2_params, "Stage 2")
        self.best_models['stage2'] = stage2_model
        
        cv_scores = cross_val_score(stage2_model, X_stage2, y_stage2, cv=3, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 2 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 3: Interview vs Accept
        print("\n### STAGE 3: Interview vs Accept ###")
        mask_stage3 = y > 15
        X_stage3 = X[mask_stage3]
        y_stage3 = (y[mask_stage3] >= 23).astype(int)
        
        stage3_params = self.optimize_stage_fast(X_stage3, y_stage3, "Stage 3")
        self.best_params['stage3'] = stage3_params
        
        stage3_model = self.train_ensemble_stage(X_stage3, y_stage3, stage3_params, "Stage 3")
        self.best_models['stage3'] = stage3_model
        
        cv_scores = cross_val_score(stage3_model, X_stage3, y_stage3, cv=3, scoring='roc_auc', n_jobs=self.n_jobs)
        print(f"Stage 3 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
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
        
        # Enhanced confidence calculation
        stage1_confidence = np.maximum(prob_not_reject, 1 - prob_not_reject)
        
        # Apply sigmoid transformation for better calibration
        stage1_confidence = 1 / (1 + np.exp(-10 * (stage1_confidence - 0.5)))
        
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
            stage2_confidence = np.maximum(prob_higher, 1 - prob_higher)
            stage2_confidence = 1 / (1 + np.exp(-10 * (stage2_confidence - 0.5)))
            
            # Waitlist decisions
            is_waitlist = prob_higher < 0.5
            waitlist_indices = np.where(non_reject_mask)[0][is_waitlist]
            predictions[waitlist_indices] = 1
            probabilities[waitlist_indices, 1] = 1 - prob_higher[is_waitlist]
            
            # Combine confidences using weighted average
            non_reject_conf = confidences[non_reject_mask]
            non_reject_conf[is_waitlist] = (
                0.6 * stage1_confidence[non_reject_mask][is_waitlist] + 
                0.4 * stage2_confidence[is_waitlist]
            )
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
                stage3_confidence = np.maximum(prob_accept, 1 - prob_accept)
                stage3_confidence = 1 / (1 + np.exp(-10 * (stage3_confidence - 0.5)))
                
                # Accept/Interview decisions
                is_accept = prob_accept >= 0.5
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                predictions[accept_indices] = 3
                predictions[interview_indices] = 2
                
                probabilities[accept_indices, 3] = prob_accept[is_accept]
                probabilities[interview_indices, 2] = 1 - prob_accept[~is_accept]
                
                # Final confidence (weighted average)
                higher_conf = confidences[higher_mask]
                higher_conf = (
                    0.4 * stage1_confidence[higher_mask] + 
                    0.3 * stage2_confidence[~is_waitlist] + 
                    0.3 * stage3_confidence
                )
                confidences[higher_mask] = higher_conf
        
        # Fill in remaining probabilities
        for i in range(n_samples):
            prob_sum = probabilities[i].sum()
            if prob_sum < 0.99:
                probabilities[i, predictions[i]] = 1 - prob_sum
        
        # Convert confidence to 0-100 scale
        confidences = confidences * 100
        
        return predictions, probabilities, confidences
    
    def evaluate_on_test(self, df_test, feature_cols, imputer, scaler):
        """Evaluate optimized model on test data."""
        print("\n" + "="*80)
        print("EVALUATING ON 2024 TEST DATA")
        print("="*80)
        
        # Engineer same features
        df_test_enhanced = self.engineer_key_features(df_test)
        
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
        
        return predictions, confidences, probabilities, exact_match, adjacent
    
    def analyze_feature_importance(self):
        """Analyze feature importance across stages."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get max features
        max_features = max(len(imp) for imp in self.feature_importance.values())
        
        # Combine importance
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
        
        print("\nTop 20 Most Important Features:")
        for idx, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:40} {row['importance']:.4f}")
        
        return importance_df


def main():
    # Initialize optimizer
    optimizer = FastCascadeOptimizer()
    
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
    predictions, confidences, probabilities, exact_match, adjacent = optimizer.evaluate_on_test(
        df_test, feature_cols, imputer, scaler
    )
    
    # Analyze feature importance
    importance_df = optimizer.analyze_feature_importance()
    
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
        'feature_importance': importance_df
    }
    
    model_path = f"models/fast_optimized_cascade_{timestamp}.pkl"
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
    
    results_path = f"fast_refined_predictions_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}")
    
    print("\n" + "="*80)
    print("FAST OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()