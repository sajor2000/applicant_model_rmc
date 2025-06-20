"""
Integrated Training Pipeline for Medical Admissions
==================================================

This script combines structured features with LLM scores to train
a comprehensive model for predicting Application Review Scores.

Training: 2022-2023 combined data
Testing: 2024 holdout set
Model: XGBoost with 5-fold cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# Import our feature engineering pipeline
from data_transformation_pipeline import DataTransformationPipeline

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, using Random Forest instead")
    from sklearn.ensemble import RandomForestRegressor


class IntegratedTrainingPipeline:
    """
    Combines structured features and LLM scores to train the final model
    """
    
    def __init__(self):
        self.data_base_path = Path("data")
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_transformer = DataTransformationPipeline()
        
        # Target variable and threshold
        self.target_column = 'Application Review Score'
        self.interview_threshold = 19  # Score >= 19 likely gets interview
        
        # LLM feature columns
        self.llm_features = [
            'llm_narrative_coherence',
            'llm_motivation_authenticity', 
            'llm_reflection_depth',
            'llm_growth_demonstrated',
            'llm_unique_perspective',
            'llm_clinical_insight',
            'llm_service_genuineness',
            'llm_leadership_impact',
            'llm_communication_quality',
            'llm_maturity_score',
            'llm_red_flag_count',
            'llm_green_flag_count',
            'llm_overall_essay_score'
        ]
        
        print("="*80)
        print("INTEGRATED MEDICAL ADMISSIONS MODEL TRAINING")
        print("Combining Structured Features + LLM Scores")
        print("="*80)
        
    def load_and_merge_data(self, year: int, llm_scores_file: str) -> pd.DataFrame:
        """Load structured data and merge with LLM scores"""
        
        # Load structured data
        if year in [2022, 2023]:
            year_path = self.data_base_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        else:
            year_path = self.data_base_path / f"{year} Applicants Reviewed by Trusted Reviewers"
            
        structured_df = pd.read_excel(year_path / "1. Applicants.xlsx")
        
        # Standardize ID column
        if 'Amcas_ID' in structured_df.columns:
            structured_df['AMCAS_ID'] = structured_df['Amcas_ID']
        elif 'amcas_id' in structured_df.columns:
            structured_df['AMCAS_ID'] = structured_df['amcas_id']
            
        # Load LLM scores
        llm_df = pd.read_csv(llm_scores_file)
        
        # Merge on AMCAS_ID
        merged_df = structured_df.merge(
            llm_df[['AMCAS_ID'] + self.llm_features],
            on='AMCAS_ID',
            how='left'
        )
        
        # Fill missing LLM scores with median values
        for col in self.llm_features:
            if col in merged_df.columns:
                median_val = merged_df[col].median()
                merged_df[col].fillna(median_val, inplace=True)
                
        merged_df['year'] = year
        
        return merged_df
        
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare 2022-2023 training data with LLM scores"""
        print("\n[STEP 1] Loading Training Data (2022-2023)")
        print("-" * 50)
        
        # Load 2022 data with LLM scores
        df_2022 = self.load_and_merge_data(2022, "llm_scores_2022_2023_20250619_172837.csv")
        print(f"✓ Loaded 2022: {len(df_2022)} applicants")
        
        # Load 2023 data with LLM scores  
        df_2023 = self.load_and_merge_data(2023, "llm_scores_2022_2023_20250619_172837.csv")
        print(f"✓ Loaded 2023: {len(df_2023)} applicants")
        
        # Combine training data
        train_df = pd.concat([df_2022, df_2023], ignore_index=True)
        print(f"✓ Combined training data: {len(train_df)} total applicants")
        
        # Check target variable
        if self.target_column in train_df.columns:
            print(f"✓ Target variable '{self.target_column}' found")
            print(f"  Score distribution: {train_df[self.target_column].describe()}")
            print(f"  Interview rate (≥19): {(train_df[self.target_column] >= self.interview_threshold).mean():.1%}")
        else:
            raise ValueError(f"Target column '{self.target_column}' not found!")
            
        return train_df
        
    def prepare_test_data(self) -> pd.DataFrame:
        """Load and prepare 2024 test data with LLM scores"""
        print("\n[STEP 2] Loading Test Data (2024)")
        print("-" * 50)
        
        # Find the latest 2024 LLM scores file
        try:
            with open("latest_2024_llm_scores.txt", "r") as f:
                llm_file = f.read().strip()
        except:
            # Fallback to pattern matching
            llm_file = "llm_scores_2024_20250619_175241.csv"
            
        test_df = self.load_and_merge_data(2024, llm_file)
        print(f"✓ Loaded 2024 test data: {len(test_df)} applicants")
        
        if self.target_column in test_df.columns:
            print(f"  Score distribution: {test_df[self.target_column].describe()}")
            print(f"  Interview rate (≥19): {(test_df[self.target_column] >= self.interview_threshold).mean():.1%}")
            
        return test_df
        
    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply feature engineering to combined data"""
        
        # Keep a copy of LLM scores and other important columns before transformation
        llm_scores_df = None
        if all(col in df.columns for col in self.llm_features):
            llm_cols = ['AMCAS_ID'] + self.llm_features
            llm_scores_df = df[llm_cols].copy()
        
        # Also keep target and year
        target_series = df[self.target_column].copy() if self.target_column in df.columns else None
        year_series = df['year'].copy() if 'year' in df.columns else None
        amcas_ids = df['AMCAS_ID'].copy()
        
        # Process through stages (this modifies df)
        df = self.feature_transformer.stage2_handle_missing(df)
        df = self.feature_transformer.stage3_create_ratios(df)
        df = self.feature_transformer.stage4_encode_categoricals(df, fit=fit)
        
        # Final scaling with LLM scores
        X_scaled, feature_names = self.feature_transformer.stage5_select_and_scale(
            df, 
            llm_scores=llm_scores_df,
            fit_scaler=fit
        )
        
        # Add back AMCAS_ID and target if present
        X_scaled['AMCAS_ID'] = amcas_ids
        if target_series is not None:
            X_scaled[self.target_column] = target_series
        if year_series is not None:
            X_scaled['year'] = year_series
            
        return X_scaled
            
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model with 5-fold cross-validation"""
        print("\n[STEP 3] Training Model")
        print("-" * 50)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        if XGBOOST_AVAILABLE:
            print("✓ Using XGBoost Regressor")
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            print("✓ Using Random Forest Regressor")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            
        # 5-fold cross-validation with stratification on binned scores
        print("\n✓ Performing 5-fold cross-validation...")
        
        # Create stratification bins based on score ranges
        score_bins = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Track CV scores
        cv_maes = []
        cv_rmses = []
        cv_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, score_bins)):
            X_fold_train = X_train_scaled[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train on fold
            self.model.fit(X_fold_train, y_fold_train)
            
            # Predict on validation fold
            y_pred = self.model.predict(X_fold_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_fold_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            r2 = r2_score(y_fold_val, y_pred)
            
            cv_maes.append(mae)
            cv_rmses.append(rmse)
            cv_r2s.append(r2)
            
            print(f"  Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
        print(f"\n✓ Cross-validation results:")
        print(f"  MAE:  {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f}")
        print(f"  RMSE: {np.mean(cv_rmses):.3f} ± {np.std(cv_rmses):.3f}")
        print(f"  R²:   {np.mean(cv_r2s):.3f} ± {np.std(cv_r2s):.3f}")
        
        # Train final model on all training data
        print("\n✓ Training final model on full training set...")
        self.model.fit(X_train_scaled, y_train)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.analyze_feature_importance(X_train.columns)
            
    def analyze_feature_importance(self, feature_names: List[str]) -> None:
        """Analyze and display feature importance"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n✓ Top 20 Most Important Features:")
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")
            
        # Separate LLM vs structured feature importance
        llm_importance = sum(importances[i] for i, name in enumerate(feature_names) 
                           if name in self.llm_features)
        total_importance = sum(importances)
        
        print(f"\n✓ Feature Group Importance:")
        print(f"  LLM features:        {llm_importance/total_importance:.1%}")
        print(f"  Structured features: {(total_importance-llm_importance)/total_importance:.1%}")
        
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance on holdout test set"""
        print("\n[STEP 4] Evaluating on 2024 Holdout Test Set")
        print("-" * 50)
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict scores
        y_pred = self.model.predict(X_test_scaled)
        
        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Regression Performance:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")
        
        # Classification metrics at threshold 19
        y_true_binary = (y_test >= self.interview_threshold).astype(int)
        y_pred_binary = (y_pred >= self.interview_threshold).astype(int)
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary'
        )
        
        print(f"\n✓ Interview Decision Performance (≥{self.interview_threshold}):")
        print(f"  Accuracy:  {accuracy:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}")
        print(f"  F1-Score:  {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        print(f"\n✓ Confusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]} (correctly rejected)")
        print(f"  False Positives: {cm[0,1]} (incorrectly invited)")
        print(f"  False Negatives: {cm[1,0]} (incorrectly rejected)")
        print(f"  True Positives:  {cm[1,1]} (correctly invited)")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
    def save_model(self, metrics: Dict) -> str:
        """Save trained model and components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_transformer': self.feature_transformer,
            'feature_names': self.feature_names,
            'llm_features': self.llm_features,
            'target_column': self.target_column,
            'interview_threshold': self.interview_threshold,
            'training_metrics': metrics,
            'timestamp': timestamp
        }
        
        model_path = self.models_path / f"integrated_model_{timestamp}.pkl"
        joblib.dump(model_data, model_path)
        
        # Also save as 'latest' for easy access
        latest_path = self.models_path / "integrated_model_latest.pkl"
        joblib.dump(model_data, latest_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Also saved as: {latest_path}")
        
        return str(model_path)
        
    def run_pipeline(self) -> str:
        """Execute the complete training pipeline"""
        
        # Load training data
        train_df = self.prepare_training_data()
        
        # Load test data
        test_df = self.prepare_test_data()
        
        # Engineer features for training data
        print("\n✓ Engineering features for training data...")
        X_train = self.engineer_features(train_df, fit=True)
        
        # Remove target and non-feature columns
        non_feature_cols = [self.target_column, 'AMCAS_ID', 'year', 'Amcas_ID', 'amcas_id']
        feature_cols = [col for col in X_train.columns if col not in non_feature_cols]
        
        X_train = X_train[feature_cols]
        y_train = train_df[self.target_column]
        
        # Store feature names
        self.feature_names = feature_cols
        print(f"✓ Total features: {len(feature_cols)}")
        print(f"  - Structured features: {len([f for f in feature_cols if f not in self.llm_features])}")
        print(f"  - LLM features: {len([f for f in feature_cols if f in self.llm_features])}")
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Engineer features for test data
        print("\n✓ Engineering features for test data...")
        X_test = self.engineer_features(test_df, fit=False)
        X_test = X_test[feature_cols]
        y_test = test_df[self.target_column]
        
        # Evaluate on test set
        metrics = self.evaluate_on_test_set(X_test, y_test)
        
        # Save model
        model_path = self.save_model(metrics)
        
        print("\n" + "="*80)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*80)
        
        return model_path


def main():
    """Execute the integrated training pipeline"""
    pipeline = IntegratedTrainingPipeline()
    model_path = pipeline.run_pipeline()
    return model_path


if __name__ == "__main__":
    main()