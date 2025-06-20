#!/usr/bin/env python3
"""
Final GPT-4o Optimization - Refined Approach
===========================================

Focus on practical improvements to achieve >82% accuracy.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src directory
sys.path.append(str(Path(__file__).parent.parent / "src"))
from feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RefinedCascadeClassifier:
    """Refined cascade classifier focusing on quartile prediction."""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        
    def _create_quartile_labels(self, scores):
        """Convert application scores to quartile labels."""
        # Q4: 0-9 (Reject)
        # Q3: 10-15 (Waitlist)
        # Q2: 16-22 (Interview)
        # Q1: 23-25 (Accept)
        quartiles = np.zeros(len(scores))
        quartiles[scores <= 9] = 0  # Q4
        quartiles[(scores > 9) & (scores <= 15)] = 1  # Q3
        quartiles[(scores > 15) & (scores <= 22)] = 2  # Q2
        quartiles[scores > 22] = 3  # Q1
        return quartiles.astype(int)
    
    def _optimize_hyperparameters(self, X, y, stage_name, n_trials=100):
        """Optimize XGBoost hyperparameters."""
        logger.info(f"Optimizing {stage_name}...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                xgb.XGBClassifier(**params),
                X, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        logger.info(f"Best {stage_name} CV accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def fit(self, X, y):
        """Fit the cascade classifier."""
        logger.info("Training Refined Cascade Classifier")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create cascade stages
        # Stage 1: Reject vs Others
        stage1_y = (y > 9).astype(int)
        
        # Stage 2: Among non-rejects, Waitlist vs Interview+
        stage2_mask = y > 9
        stage2_y = np.zeros(len(y))
        stage2_y[stage2_mask] = (y[stage2_mask] > 15).astype(int)
        
        # Stage 3: Among Interview+, Interview vs Accept
        stage3_mask = y > 15
        stage3_y = np.zeros(len(y))
        stage3_y[stage3_mask] = (y[stage3_mask] > 22).astype(int)
        
        # Train each stage
        stages = [
            ('stage1_reject', X_scaled, stage1_y),
            ('stage2_waitlist', X_scaled[stage2_mask], stage2_y[stage2_mask]),
            ('stage3_interview', X_scaled[stage3_mask], stage3_y[stage3_mask])
        ]
        
        for stage_name, X_stage, y_stage in stages:
            logger.info(f"\nTraining {stage_name}")
            logger.info(f"  Samples: {len(y_stage)}, Positive class: {np.mean(y_stage):.2%}")
            
            # Optimize hyperparameters
            best_params = self._optimize_hyperparameters(X_stage, y_stage, stage_name)
            self.best_params[stage_name] = best_params
            
            # Train final model
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_stage, y_stage)
            self.models[stage_name] = model
            
            # Report performance
            train_pred = model.predict(X_stage)
            train_acc = accuracy_score(y_stage, train_pred)
            logger.info(f"  Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict_proba_cascade(self, X):
        """Get probability estimates through the cascade."""
        X_scaled = self.scaler.transform(X)
        n_samples = len(X)
        
        # Initialize results
        final_probs = np.zeros((n_samples, 4))  # 4 quartiles
        
        # Stage 1: Reject probability
        stage1_proba = self.models['stage1_reject'].predict_proba(X_scaled)
        reject_prob = stage1_proba[:, 0]
        final_probs[:, 0] = reject_prob
        
        # For non-rejects
        non_reject_prob = stage1_proba[:, 1]
        non_reject_mask = non_reject_prob > 0.5
        
        if np.any(non_reject_mask):
            # Stage 2: Waitlist vs Interview+
            stage2_proba = self.models['stage2_waitlist'].predict_proba(X_scaled[non_reject_mask])
            waitlist_prob = stage2_proba[:, 0] * non_reject_prob[non_reject_mask]
            interview_plus_prob = stage2_proba[:, 1] * non_reject_prob[non_reject_mask]
            
            final_probs[non_reject_mask, 1] = waitlist_prob
            
            # Stage 3: Interview vs Accept
            interview_plus_indices = np.where(non_reject_mask)[0][stage2_proba[:, 1] > 0.5]
            
            if len(interview_plus_indices) > 0:
                stage3_proba = self.models['stage3_interview'].predict_proba(X_scaled[interview_plus_indices])
                interview_prob = stage3_proba[:, 0] * interview_plus_prob[stage2_proba[:, 1] > 0.5]
                accept_prob = stage3_proba[:, 1] * interview_plus_prob[stage2_proba[:, 1] > 0.5]
                
                final_probs[interview_plus_indices, 2] = interview_prob
                final_probs[interview_plus_indices, 3] = accept_prob
        
        return final_probs
    
    def predict(self, X):
        """Predict quartile labels."""
        probs = self.predict_proba_cascade(X)
        return np.argmax(probs, axis=1)


def engineer_features(X, feature_names):
    """Add engineered features to improve performance."""
    X_df = pd.DataFrame(X, columns=feature_names)
    new_features = []
    new_names = []
    
    # 1. LLM score statistics
    llm_cols = [col for col in feature_names if col.startswith('llm_') and 'count' not in col]
    if llm_cols:
        llm_mean = X_df[llm_cols].mean(axis=1)
        llm_std = X_df[llm_cols].std(axis=1)
        llm_max = X_df[llm_cols].max(axis=1)
        llm_min = X_df[llm_cols].min(axis=1)
        
        new_features.extend([
            llm_mean.values.reshape(-1, 1),
            llm_std.values.reshape(-1, 1),
            llm_max.values.reshape(-1, 1),
            llm_min.values.reshape(-1, 1)
        ])
        new_names.extend(['llm_mean', 'llm_std', 'llm_max', 'llm_min'])
    
    # 2. Academic performance indicators
    if 'Undergrad_GPA' in feature_names and 'Undergrad_BCPM' in feature_names:
        gpa_diff = X_df['Undergrad_GPA'] - X_df['Undergrad_BCPM']
        new_features.append(gpa_diff.values.reshape(-1, 1))
        new_names.append('gpa_bcpm_diff')
    
    # 3. Experience diversity
    exp_cols = [col for col in feature_names if 'Exp_Hour' in col or 'Hours' in col]
    if exp_cols:
        exp_diversity = (X_df[exp_cols] > 0).sum(axis=1)
        new_features.append(exp_diversity.values.reshape(-1, 1))
        new_names.append('experience_diversity')
    
    # 4. Flag balance
    if 'llm_red_flag_count' in feature_names and 'llm_green_flag_count' in feature_names:
        flag_balance = X_df['llm_green_flag_count'] - X_df['llm_red_flag_count']
        new_features.append(flag_balance.values.reshape(-1, 1))
        new_names.append('flag_balance')
    
    if new_features:
        X_enhanced = np.hstack([X] + new_features)
        enhanced_names = list(feature_names) + new_names
        logger.info(f"Added {len(new_names)} engineered features")
        return X_enhanced, enhanced_names
    
    return X, feature_names


def evaluate_model(model, X, y, feature_names):
    """Evaluate model performance."""
    # Convert scores to quartiles
    y_quartiles = model._create_quartile_labels(y)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_quartiles, y_pred)
    
    # Adjacent accuracy (within 1 quartile)
    adjacent = np.abs(y_pred - y_quartiles) <= 1
    adjacent_acc = np.mean(adjacent)
    
    # Confusion matrix
    cm = confusion_matrix(y_quartiles, y_pred)
    
    # Confidence scores
    probs = model.predict_proba_cascade(X)
    confidences = np.max(probs, axis=1)
    
    logger.info("\nModel Evaluation:")
    logger.info(f"  Exact Match Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"  Adjacent Accuracy: {adjacent_acc:.3f} ({adjacent_acc*100:.1f}%)")
    logger.info(f"  Mean Confidence: {np.mean(confidences):.3f}")
    
    # Per-quartile performance
    for i in range(4):
        mask = y_quartiles == i
        if np.any(mask):
            quartile_acc = accuracy_score(y_quartiles[mask], y_pred[mask])
            logger.info(f"  Q{4-i} Accuracy: {quartile_acc:.3f}")
    
    return {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_acc,
        'confusion_matrix': cm,
        'confidences': confidences,
        'predictions': y_pred,
        'true_quartiles': y_quartiles
    }


def create_visualizations(results, output_dir):
    """Create result visualizations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Q4', 'Q3', 'Q2', 'Q1'],
                yticklabels=['Q4', 'Q3', 'Q2', 'Q1'])
    plt.title('Refined GPT-4o Model Confusion Matrix')
    plt.ylabel('True Quartile')
    plt.xlabel('Predicted Quartile')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['confidences'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(results['confidences']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(results["confidences"]):.2f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_dist.png", dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main optimization function."""
    logger.info("🚀 STARTING REFINED GPT-4O OPTIMIZATION")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading training data...")
    train_dfs = []
    
    for year in ['2022', '2023']:
        # Load structured data
        df_struct = pd.read_excel(f"data/{year} Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
        
        # Load LLM scores
        df_llm = pd.read_csv(f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv")
        
        # Standardize AMCAS ID
        if 'Amcas_ID' in df_struct.columns:
            df_struct['AMCAS ID'] = df_struct['Amcas_ID'].astype(str)
        df_llm['AMCAS ID'] = df_llm['AMCAS ID'].astype(str)
        
        # Merge
        df_merged = pd.merge(df_struct, df_llm, on='AMCAS ID', how='inner')
        train_dfs.append(df_merged)
        logger.info(f"  {year}: {len(df_merged)} records")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    logger.info(f"Total training records: {len(train_df)}")
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(train_df)
    
    # Add engineered features
    X_enhanced, feature_names = engineer_features(X, engineer.feature_names)
    
    logger.info(f"Final feature matrix: {X_enhanced.shape}")
    
    # Train model
    model = RefinedCascadeClassifier()
    model.fit(X_enhanced, y)
    
    # Evaluate
    results = evaluate_model(model, X_enhanced, y, feature_names)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_data = {
        'model': model,
        'feature_engineer': engineer,
        'feature_names': feature_names,
        'results': results,
        'timestamp': timestamp
    }
    
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/refined_gpt4o_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    joblib.dump(model_data, "models/refined_gpt4o_latest.pkl")
    logger.info(f"Model saved to {model_path}")
    
    # Create visualizations
    create_visualizations(results, f"output/refined_{timestamp}")
    
    logger.info("=" * 60)
    logger.info("🏁 OPTIMIZATION COMPLETE!")
    logger.info(f"Final Accuracy: {results['accuracy']:.1%}")
    logger.info(f"Adjacent Accuracy: {results['adjacent_accuracy']:.1%}")
    
    return model_data


if __name__ == "__main__":
    main()