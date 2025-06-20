"""
Comprehensive Model Refinement - XGBoost Focus
==============================================

Implements extensive hyperparameter tuning, ensemble methods,
and advanced techniques using XGBoost as the primary algorithm.
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
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    make_scorer, cohen_kappa_score, accuracy_score,
    precision_recall_curve, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Custom scorer for adjacent accuracy
def adjacent_accuracy_scorer(y_true, y_pred):
    """Score based on predictions being within 1 bucket."""
    return np.mean(np.abs(y_true - y_pred) <= 1)

adjacent_scorer = make_scorer(adjacent_accuracy_scorer, greater_is_better=True)


class RefinedCascadeOptimizer:
    """Refined optimization for cascade classifier."""
    
    def __init__(self):
        self.best_params = {}
        self.best_models = {}
        self.feature_importance = {}
        
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
    
    def engineer_confidence_features(self, df):
        """Create features specifically for improving confidence."""
        print("Engineering confidence-enhancing features...")
        
        df = df.copy()
        
        # 1. Profile Completeness Score
        important_fields = [
            'healthcare_total_hours', 'exp_hour_research', 'service_rating_numerical',
            'exp_hour_volunteer_med', 'age', 'first_generation_ind'
        ]
        df['profile_completeness'] = 0
        for field in important_fields:
            if field in df.columns:
                df['profile_completeness'] += (~df[field].isna()).astype(int)
        df['profile_completeness'] = df['profile_completeness'] / len(important_fields)
        
        # 2. Experience Consistency
        exp_cols = ['exp_hour_research', 'exp_hour_volunteer_med', 
                   'exp_hour_volunteer_non_med', 'exp_hour_shadowing']
        available_exp = [col for col in exp_cols if col in df.columns]
        
        if len(available_exp) > 1:
            # Coefficient of variation - lower means more consistent
            exp_mean = df[available_exp].mean(axis=1)
            exp_std = df[available_exp].std(axis=1)
            df['experience_consistency'] = 1 / (1 + exp_std / (exp_mean + 1))
            
            # Experience diversity (number of different activities)
            df['experience_diversity'] = (df[available_exp] > 50).sum(axis=1)
        
        # 3. Essay-Structure Alignment
        if 'llm_overall_essay_score' in df.columns and 'service_rating_numerical' in df.columns:
            # Normalize both to 0-1 scale
            essay_norm = df['llm_overall_essay_score'] / 100
            service_norm = (df['service_rating_numerical'] - 1) / 3  # 1-4 scale to 0-1
            
            # Alignment score - closer values = higher alignment
            df['essay_service_alignment'] = 1 - abs(essay_norm - service_norm)
        
        # 4. LLM Score Consistency
        llm_score_cols = [col for col in df.columns if col.startswith('llm_') and 
                         col not in ['llm_red_flag_count', 'llm_green_flag_count']]
        if len(llm_score_cols) > 2:
            # Normalize LLM scores to 0-1
            for col in llm_score_cols:
                df[f'{col}_norm'] = df[col] / df[col].max()
            
            norm_cols = [f'{col}_norm' for col in llm_score_cols]
            df['llm_consistency'] = 1 / (1 + df[norm_cols].std(axis=1))
            
            # Drop normalized columns
            df = df.drop(columns=norm_cols)
        
        # 5. Extreme Value Indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['extreme_value_count'] = 0
        
        for col in numeric_cols:
            if col not in ['amcas_id', 'application_review_score', 'extreme_value_count']:
                p95 = df[col].quantile(0.95)
                p5 = df[col].quantile(0.05)
                df['extreme_value_count'] += ((df[col] > p95) | (df[col] < p5)).astype(int)
        
        # 6. Red/Green Flag Balance
        if 'llm_red_flag_count' in df.columns and 'llm_green_flag_count' in df.columns:
            df['flag_balance'] = df['llm_green_flag_count'] - df['llm_red_flag_count']
            df['flag_ratio'] = df['llm_green_flag_count'] / (df['llm_red_flag_count'] + 1)
        
        # 7. Key Interaction Features
        if 'service_rating_numerical' in df.columns:
            if 'healthcare_total_hours' in df.columns:
                df['service_x_clinical'] = (
                    df['service_rating_numerical'] * np.log1p(df['healthcare_total_hours'])
                )
            
            if 'llm_overall_essay_score' in df.columns:
                df['service_x_essay'] = (
                    df['service_rating_numerical'] * df['llm_overall_essay_score'] / 25
                )
        
        # 8. Polynomial features for key predictors
        key_features = ['service_rating_numerical', 'healthcare_total_hours']
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
                df[f'{feat}_sqrt'] = np.sqrt(df[feat].clip(lower=0))
        
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
    
    def grid_search_stage(self, X, y, stage_name):
        """Extensive grid search for stage optimization."""
        print(f"\nOptimizing {stage_name} with grid search...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [500, 700, 900],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.5, 1, 2],
            'reg_lambda': [1, 2, 3],
            'gamma': [0, 0.1, 0.3]
        }
        
        # Use RandomizedSearchCV for efficiency
        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'  # Faster training
        )
        
        # Calculate positive class weight
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        
        random_search = RandomizedSearchCV(
            xgb,
            param_distributions=param_grid,
            n_iter=100,  # Try 100 random combinations
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        best_params = random_search.best_params_
        best_params['scale_pos_weight'] = pos_weight
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        
        print(f"Best AUC for {stage_name}: {random_search.best_score_:.4f}")
        print(f"Best params: {best_params}")
        
        return best_params
    
    def train_ensemble_stage(self, X, y, stage_params, stage_name):
        """Train ensemble of XGBoost models with different seeds."""
        print(f"\nTraining ensemble for {stage_name}...")
        
        models = []
        
        # Train 5 models with different random seeds
        for i in range(5):
            params = stage_params.copy()
            params['random_state'] = i * 42
            
            # Slightly vary some parameters
            params['n_estimators'] = int(params['n_estimators'] * (1 + i * 0.05))
            params['max_depth'] = params['max_depth'] + (i % 2)
            params['learning_rate'] = params['learning_rate'] * (1 + i * 0.02)
            
            model = XGBClassifier(**params)
            model.fit(X, y)
            models.append(model)
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[(f'xgb_{i}', model) for i, model in enumerate(models)],
            voting='soft'
        )
        
        # Fit ensemble (required by sklearn even though individual models are fitted)
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
        """Train fully optimized cascade classifier."""
        print("\n" + "="*80)
        print("TRAINING OPTIMIZED CASCADE CLASSIFIER")
        print("="*80)
        
        # Engineer confidence features
        df_enhanced = self.engineer_confidence_features(df_train)
        
        # Prepare data
        X, y, feature_cols, imputer, scaler = self.prepare_cascade_data(df_enhanced)
        
        # Save feature columns for later analysis
        self.feature_cols = feature_cols
        
        # Stage 1: Reject vs Non-Reject
        print("\n### STAGE 1: Reject vs Non-Reject ###")
        y_stage1 = (y > 9).astype(int)
        
        # Grid search
        stage1_params = self.grid_search_stage(X, y_stage1, "Stage 1")
        self.best_params['stage1'] = stage1_params
        
        # Train ensemble
        stage1_model = self.train_ensemble_stage(X, y_stage1, stage1_params, "Stage 1")
        self.best_models['stage1'] = stage1_model
        
        # Cross-validation performance
        cv_scores = cross_val_score(stage1_model, X, y_stage1, cv=5, scoring='roc_auc')
        print(f"Stage 1 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 2: Waitlist vs Higher
        print("\n### STAGE 2: Waitlist vs Higher ###")
        mask_stage2 = y > 9
        X_stage2 = X[mask_stage2]
        y_stage2 = (y[mask_stage2] > 15).astype(int)
        
        stage2_params = self.grid_search_stage(X_stage2, y_stage2, "Stage 2")
        self.best_params['stage2'] = stage2_params
        
        stage2_model = self.train_ensemble_stage(X_stage2, y_stage2, stage2_params, "Stage 2")
        self.best_models['stage2'] = stage2_model
        
        cv_scores = cross_val_score(stage2_model, X_stage2, y_stage2, cv=5, scoring='roc_auc')
        print(f"Stage 2 CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Stage 3: Interview vs Accept
        print("\n### STAGE 3: Interview vs Accept ###")
        mask_stage3 = y > 15
        X_stage3 = X[mask_stage3]
        y_stage3 = (y[mask_stage3] >= 23).astype(int)
        
        stage3_params = self.grid_search_stage(X_stage3, y_stage3, "Stage 3")
        self.best_params['stage3'] = stage3_params
        
        stage3_model = self.train_ensemble_stage(X_stage3, y_stage3, stage3_params, "Stage 3")
        self.best_models['stage3'] = stage3_model
        
        cv_scores = cross_val_score(stage3_model, X_stage3, y_stage3, cv=5, scoring='roc_auc')
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
        
        # Calculate confidence based on probability extremity
        stage1_confidence = np.maximum(prob_not_reject, 1 - prob_not_reject)
        stage1_confidence = (stage1_confidence - 0.5) * 2  # Scale to 0-1
        
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
            stage2_confidence = (stage2_confidence - 0.5) * 2
            
            # Waitlist decisions
            is_waitlist = prob_higher < 0.5
            waitlist_indices = np.where(non_reject_mask)[0][is_waitlist]
            predictions[waitlist_indices] = 1
            probabilities[waitlist_indices, 1] = 1 - prob_higher[is_waitlist]
            
            # Combine confidences (geometric mean)
            non_reject_conf = confidences[non_reject_mask]
            non_reject_conf[is_waitlist] = np.sqrt(
                stage1_confidence[non_reject_mask][is_waitlist] * 
                stage2_confidence[is_waitlist]
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
                stage3_confidence = (stage3_confidence - 0.5) * 2
                
                # Accept/Interview decisions
                is_accept = prob_accept >= 0.5
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                predictions[accept_indices] = 3
                predictions[interview_indices] = 2
                
                probabilities[accept_indices, 3] = prob_accept[is_accept]
                probabilities[interview_indices, 2] = 1 - prob_accept[~is_accept]
                
                # Final confidence (geometric mean of all stages)
                higher_conf = confidences[higher_mask]
                higher_conf = np.cbrt(
                    stage1_confidence[higher_mask] * 
                    stage2_confidence[~is_waitlist] * 
                    stage3_confidence
                )
                confidences[higher_mask] = higher_conf
        
        # Fill in remaining probabilities
        for i in range(n_samples):
            prob_sum = probabilities[i].sum()
            if prob_sum < 0.99:  # Account for numerical errors
                probabilities[i, predictions[i]] = 1 - prob_sum
        
        # Convert confidence to 0-100 scale with better calibration
        # Use a sigmoid transformation for smoother distribution
        confidences = 100 / (1 + np.exp(-5 * (confidences - 0.5)))
        
        return predictions, probabilities, confidences
    
    def evaluate_on_test(self, df_test, feature_cols, imputer, scaler):
        """Evaluate optimized model on test data."""
        print("\n" + "="*80)
        print("EVALUATING ON 2024 TEST DATA")
        print("="*80)
        
        # Engineer same features
        df_test_enhanced = self.engineer_confidence_features(df_test)
        
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
        """Analyze and visualize feature importance."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get the maximum feature length across all stages
        max_features = max(len(imp) for imp in self.feature_importance.values())
        
        # Combine importance across stages, handling different lengths
        combined_importance = np.zeros(max_features)
        
        for stage, importance in self.feature_importance.items():
            # Pad importance array if needed
            if len(importance) < max_features:
                padded_importance = np.pad(importance, (0, max_features - len(importance)), mode='constant')
            else:
                padded_importance = importance[:max_features]
            combined_importance += padded_importance
        
        # Normalize
        combined_importance = combined_importance / combined_importance.sum()
        
        # Use only the features we have names for
        feature_cols_to_use = list(self.feature_cols[:min(len(self.feature_cols), len(combined_importance))])
        if len(feature_cols_to_use) < len(combined_importance):
            # Add generic names for extra features
            for i in range(len(feature_cols_to_use), len(combined_importance)):
                feature_cols_to_use.append(f'feature_{i}')
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_cols_to_use,
            'importance': combined_importance[:len(feature_cols_to_use)]
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        for idx, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:40} {row['importance']:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance (Combined Across Stages)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('refined_feature_importance.png', dpi=300)
        print("\nFeature importance plot saved to refined_feature_importance.png")
        
        return importance_df


def main():
    # Initialize optimizer
    optimizer = RefinedCascadeOptimizer()
    
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
    
    model_path = f"models/refined_cascade_{timestamp}.pkl"
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
    
    results_path = f"refined_predictions_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()