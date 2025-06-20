#!/usr/bin/env python3
"""
Complete GPT-4o Model Optimization
==================================

Goal: Achieve maximum performance on 2022/2023 training data
without looking at 2024 test data. Target: >82% accuracy.

Approaches:
1. Error analysis on current model
2. Advanced feature engineering  
3. Extended hyperparameter optimization
4. Ensemble methods
5. Confidence calibration
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/gpt4o_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedGPT4oCascadeClassifier:
    """
    Enhanced cascade classifier with all optimization techniques.
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_models = {}
        self.calibrators = {}
        self.best_params = {}
        self.error_analysis = {}
        
    def analyze_errors(self, X_train, y_train, X_val, y_val):
        """Analyze classification errors to identify patterns."""
        logger.info("Performing error analysis...")
        
        # Train a quick model for error analysis
        quick_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
        quick_model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = quick_model.predict(X_val)
        
        # Analyze misclassifications
        misclassified = y_pred != y_val
        error_indices = np.where(misclassified)[0]
        
        # Analyze feature importance for errors
        if len(error_indices) > 0:
            error_features = X_val[error_indices]
            correct_features = X_val[~misclassified]
            
            # Compare feature distributions
            feature_diffs = np.abs(error_features.mean(axis=0) - correct_features.mean(axis=0))
            top_error_features = np.argsort(feature_diffs)[-10:]
            
            self.error_analysis = {
                'error_rate': len(error_indices) / len(y_val),
                'error_indices': error_indices,
                'top_error_features': top_error_features,
                'feature_importance': quick_model.feature_importances_
            }
            
            logger.info(f"Error rate: {self.error_analysis['error_rate']:.2%}")
            logger.info(f"Top error-prone features: {top_error_features}")
    
    def engineer_advanced_features(self, X, feature_names):
        """Create advanced engineered features."""
        logger.info("Engineering advanced features...")
        
        X_df = pd.DataFrame(X, columns=feature_names)
        new_features = []
        new_names = []
        
        # 1. Interaction features for top LLM scores
        llm_features = [f for f in feature_names if f.startswith('llm_')]
        for i, feat1 in enumerate(llm_features[:5]):  # Top 5 LLM features
            for feat2 in llm_features[i+1:6]:
                interaction = X_df[feat1] * X_df[feat2]
                new_features.append(interaction.values.reshape(-1, 1))
                new_names.append(f'{feat1}_x_{feat2}')
        
        # 2. Ratio features
        if 'Total_GPA' in feature_names and 'BCPM_GPA' in feature_names:
            ratio = X_df['BCPM_GPA'] / (X_df['Total_GPA'] + 0.01)
            new_features.append(ratio.values.reshape(-1, 1))
            new_names.append('BCPM_to_Total_GPA_ratio')
        
        # 3. Composite scores
        if any('MCAT' in f for f in feature_names):
            mcat_cols = [f for f in feature_names if 'MCAT' in f and f != 'MCAT_Total']
            if mcat_cols:
                mcat_variance = X_df[mcat_cols].var(axis=1)
                new_features.append(mcat_variance.values.reshape(-1, 1))
                new_names.append('MCAT_subscore_variance')
        
        # 4. Essay quality indicators
        essay_scores = [f for f in feature_names if 'essay' in f or 'motivation' in f or 'clinical' in f]
        if essay_scores:
            essay_mean = X_df[essay_scores].mean(axis=1)
            essay_std = X_df[essay_scores].std(axis=1)
            new_features.append(essay_mean.values.reshape(-1, 1))
            new_features.append(essay_std.values.reshape(-1, 1))
            new_names.extend(['essay_quality_mean', 'essay_quality_consistency'])
        
        # 5. Red/Green flag ratio
        if 'llm_red_flag_count' in feature_names and 'llm_green_flag_count' in feature_names:
            flag_score = X_df['llm_green_flag_count'] - X_df['llm_red_flag_count']
            new_features.append(flag_score.values.reshape(-1, 1))
            new_names.append('net_flag_score')
        
        if new_features:
            X_enhanced = np.hstack([X] + new_features)
            enhanced_names = list(feature_names) + new_names
            logger.info(f"Added {len(new_names)} engineered features")
            return X_enhanced, enhanced_names
        
        return X, feature_names
    
    def optimize_stage_extended(self, X, y, stage_name, n_trials=200):
        """Extended hyperparameter optimization with more sophisticated search."""
        logger.info(f"Extended optimization for {stage_name} with {n_trials} trials")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            # Cross-validation with careful handling
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                pred = model.predict(X_val_cv)
                scores.append(accuracy_score(y_val_cv, pred))
            
            return np.mean(scores)
        
        # Create study with advanced pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        logger.info(f"Best {stage_name} accuracy: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def build_ensemble(self, X_train, y_train, stage_name):
        """Build ensemble of diverse models."""
        logger.info(f"Building ensemble for {stage_name}")
        
        # Base models with diversity
        models = [
            ('xgb_primary', xgb.XGBClassifier(
                **self.best_params[stage_name],
                random_state=42
            )),
            ('xgb_secondary', xgb.XGBClassifier(
                **{**self.best_params[stage_name], 
                   'subsample': 0.8,
                   'colsample_bytree': 0.8},
                random_state=123
            )),
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=self.best_params[stage_name].get('max_depth', 6),
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=self.best_params[stage_name].get('max_depth', 6) - 1,
                learning_rate=0.1,
                random_state=42
            ))
        ]
        
        # Voting classifier with soft voting
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def calibrate_predictions(self, model, X_train, y_train):
        """Calibrate model predictions for better confidence estimates."""
        logger.info("Calibrating predictions...")
        
        # Use isotonic regression for calibration
        calibrated = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=3
        )
        
        calibrated.fit(X_train, y_train)
        return calibrated
    
    def fit(self, X, y, feature_names):
        """Fit the complete optimized cascade classifier."""
        logger.info("=" * 80)
        logger.info("TRAINING OPTIMIZED GPT-4O CASCADE CLASSIFIER")
        logger.info("=" * 80)
        
        # Engineer advanced features
        X_enhanced, enhanced_names = self.engineer_advanced_features(X, feature_names)
        self.feature_names = enhanced_names
        
        # Split for error analysis
        split_idx = int(0.8 * len(X))
        X_train, X_val = X_enhanced[:split_idx], X_enhanced[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Perform error analysis
        self.analyze_errors(X_train, y_train, X_val, y_val)
        
        # Define cascade stages
        # Stage 1: Reject (0-9) vs Non-Reject (10+)
        stage1_y = (y > 9).astype(int)
        
        # Stage 2: Waitlist (10-15) vs Interview+ (16+)
        stage2_mask = y > 9
        stage2_y = np.zeros(len(y))
        stage2_y[stage2_mask] = (y[stage2_mask] > 15).astype(int)
        
        # Stage 3: Interview (16-22) vs Accept (23+)
        stage3_mask = y > 15
        stage3_y = np.zeros(len(y))
        stage3_y[stage3_mask] = (y[stage3_mask] > 22).astype(int)
        
        # Optimize each stage
        stages = [
            ('stage1', X_enhanced, stage1_y),
            ('stage2', X_enhanced[stage2_mask], stage2_y[stage2_mask]),
            ('stage3', X_enhanced[stage3_mask], stage3_y[stage3_mask])
        ]
        
        for stage_name, X_stage, y_stage in stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"OPTIMIZING {stage_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Extended hyperparameter optimization
            self.best_params[stage_name] = self.optimize_stage_extended(
                X_stage, y_stage, stage_name
            )
            
            # Build ensemble
            ensemble = self.build_ensemble(X_stage, y_stage, stage_name)
            
            # Calibrate predictions
            calibrated = self.calibrate_predictions(ensemble, X_stage, y_stage)
            
            self.models[stage_name] = calibrated
            
            # Evaluate stage performance
            cv_pred = cross_val_predict(calibrated, X_stage, y_stage, cv=5)
            stage_acc = accuracy_score(y_stage, cv_pred)
            logger.info(f"{stage_name} cross-val accuracy: {stage_acc:.4f}")
        
        return self
    
    def predict_with_confidence(self, X):
        """Make predictions with calibrated confidence scores."""
        # Engineer features
        X_enhanced, _ = self.engineer_advanced_features(X, self.feature_names[:X.shape[1]])
        
        predictions = np.zeros(len(X))
        confidences = np.zeros(len(X))
        
        # Stage 1
        stage1_proba = self.models['stage1'].predict_proba(X_enhanced)[:, 1]
        stage1_pred = stage1_proba > 0.5
        
        # Reject (Q4)
        reject_mask = ~stage1_pred
        predictions[reject_mask] = 1  # Q4
        confidences[reject_mask] = 1 - stage1_proba[reject_mask]
        
        # Stage 2 for non-rejects
        if np.any(stage1_pred):
            stage2_proba = self.models['stage2'].predict_proba(X_enhanced[stage1_pred])[:, 1]
            stage2_pred = stage2_proba > 0.5
            
            # Waitlist (Q3)
            waitlist_indices = np.where(stage1_pred)[0][~stage2_pred]
            predictions[waitlist_indices] = 2  # Q3
            confidences[waitlist_indices] = 1 - stage2_proba[~stage2_pred]
            
            # Stage 3 for interview+
            if np.any(stage2_pred):
                higher_indices = np.where(stage1_pred)[0][stage2_pred]
                stage3_proba = self.models['stage3'].predict_proba(X_enhanced[higher_indices])[:, 1]
                stage3_pred = stage3_proba > 0.5
                
                # Interview (Q2) and Accept (Q1)
                interview_indices = higher_indices[~stage3_pred]
                accept_indices = higher_indices[stage3_pred]
                
                predictions[interview_indices] = 3  # Q2
                predictions[accept_indices] = 4  # Q1
                
                confidences[interview_indices] = 1 - stage3_proba[~stage3_pred]
                confidences[accept_indices] = stage3_proba[stage3_pred]
        
        return predictions.astype(int), confidences


def load_data():
    """Load 2022 and 2023 training data only."""
    logger.info("Loading 2022 and 2023 training data...")
    
    train_dfs = []
    
    for year in ['2022', '2023']:
        # Load structured data
        applicants_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
        df_struct = pd.read_excel(applicants_path)
        
        # Load LLM scores (GPT-4o)
        llm_scores_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv"
        
        if not Path(llm_scores_path).exists():
            logger.warning(f"LLM scores not found for {year}, skipping...")
            continue
            
        df_llm = pd.read_csv(llm_scores_path)
        
        # Standardize AMCAS ID - handle different column names
        if 'AMCAS ID' in df_struct.columns:
            df_struct['AMCAS ID'] = df_struct['AMCAS ID'].astype(str)
        elif 'Amcas_ID' in df_struct.columns:
            df_struct['AMCAS ID'] = df_struct['Amcas_ID'].astype(str)
        
        df_llm['AMCAS ID'] = df_llm['AMCAS ID'].astype(str)
        
        # Merge
        df_merged = pd.merge(df_struct, df_llm, on='AMCAS ID', how='inner')
        df_merged['year'] = year
        train_dfs.append(df_merged)
        
        logger.info(f"Loaded {len(df_merged)} records for {year}")
    
    # Combine training data
    train_df = pd.concat(train_dfs, ignore_index=True)
    logger.info(f"Total training records: {len(train_df)}")
    
    return train_df


def evaluate_model(model, X, y, feature_names):
    """Comprehensive model evaluation."""
    predictions, confidences = model.predict_with_confidence(X)
    
    # Convert to quartiles
    def score_to_quartile(score):
        if score <= 9: return 1  # Q4
        elif score <= 15: return 2  # Q3
        elif score <= 22: return 3  # Q2
        else: return 4  # Q1
    
    y_quartiles = np.array([score_to_quartile(s) for s in y])
    
    # Metrics
    exact_match = accuracy_score(y_quartiles, predictions)
    
    # Adjacent accuracy
    adjacent = np.abs(predictions - y_quartiles) <= 1
    adjacent_acc = np.mean(adjacent)
    
    # Confidence analysis
    mean_conf = np.mean(confidences)
    high_conf = np.mean(confidences > 0.8)
    low_conf = np.mean(confidences < 0.6)
    
    # Confusion matrix
    cm = confusion_matrix(y_quartiles, predictions, labels=[1, 2, 3, 4])
    
    results = {
        'exact_accuracy': exact_match,
        'adjacent_accuracy': adjacent_acc,
        'mean_confidence': mean_conf,
        'high_confidence_pct': high_conf,
        'low_confidence_pct': low_conf,
        'confusion_matrix': cm,
        'predictions': predictions,
        'confidences': confidences,
        'true_quartiles': y_quartiles
    }
    
    logger.info(f"\nEVALUATION RESULTS:")
    logger.info(f"Exact Match Accuracy: {exact_match:.3f} ({exact_match*100:.1f}%)")
    logger.info(f"Adjacent Accuracy: {adjacent_acc:.3f} ({adjacent_acc*100:.1f}%)")
    logger.info(f"Mean Confidence: {mean_conf:.3f}")
    logger.info(f"High Confidence (>80%): {high_conf:.1%}")
    logger.info(f"Low Confidence (<60%): {low_conf:.1%}")
    
    return results


def create_visualizations(results, output_dir):
    """Create comprehensive visualizations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Q4', 'Q3', 'Q2', 'Q1'],
                yticklabels=['Q4', 'Q3', 'Q2', 'Q1'])
    plt.title('Optimized GPT-4o Model Confusion Matrix')
    plt.ylabel('True Quartile')
    plt.xlabel('Predicted Quartile')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence Distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(results['confidences'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(results['confidences']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(results["confidences"]):.2f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Confidence by accuracy
    correct = results['predictions'] == results['true_quartiles']
    plt.boxplot([results['confidences'][correct], results['confidences'][~correct]],
                labels=['Correct', 'Incorrect'])
    plt.ylabel('Confidence Score')
    plt.title('Confidence by Prediction Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main optimization function."""
    logger.info("🚀 STARTING GPT-4O COMPLETE OPTIMIZATION")
    logger.info("=" * 80)
    
    # Load training data (2022 + 2023 only)
    train_df = load_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    X_train, y_train = engineer.fit_transform(train_df)
    feature_names = engineer.feature_names
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    # Train optimized model
    model = OptimizedGPT4oCascadeClassifier()
    model.fit(X_train, y_train, feature_names)
    
    # Evaluate on training data (cross-validation already done internally)
    results = evaluate_model(model, X_train, y_train, feature_names)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_data = {
        'model': model,
        'feature_engineer': engineer,
        'feature_names': model.feature_names,
        'results': results,
        'timestamp': timestamp,
        'training_years': ['2022', '2023']
    }
    
    model_path = f"models/optimized_gpt4o_cascade_{timestamp}.pkl"
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save as latest
    joblib.dump(model_data, "models/optimized_gpt4o_cascade_latest.pkl")
    
    # Create visualizations
    create_visualizations(results, f"output/optimized_gpt4o_{timestamp}")
    
    logger.info("=" * 80)
    logger.info("🏁 OPTIMIZATION COMPLETE!")
    logger.info(f"Final Exact Accuracy: {results['exact_accuracy']:.1%}")
    logger.info(f"Final Adjacent Accuracy: {results['adjacent_accuracy']:.1%}")
    logger.info("=" * 80)
    
    return model_data


if __name__ == "__main__":
    main()