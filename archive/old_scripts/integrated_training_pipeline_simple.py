"""
Simplified Integrated Training Pipeline for Medical Admissions
=============================================================

This script combines structured features with LLM scores using a 
simpler approach to avoid complex transformations.

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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, using Random Forest instead")
    from sklearn.ensemble import RandomForestRegressor


class SimpleIntegratedPipeline:
    """
    Simplified pipeline that focuses on key features + LLM scores
    """
    
    def __init__(self):
        self.data_base_path = Path("data")
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Target variable and threshold
        self.target_column = 'Application Review Score'
        self.interview_threshold = 19
        
        # Key structured features
        self.structured_features = [
            # Academic
            'Total_GPA', 'BCPM_GPA', 'Age',
            
            # Experience hours
            'Exp_Hour_Total', 'Exp_Hour_Research', 
            'Exp_Hour_Volunteer_Med', 'Exp_Hour_Volunteer_Non_Med',
            'Exp_Hour_Employ_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            
            # Diversity indicators
            'First_Generation_Ind', 'Disadvantanged_Ind', 
            'RU_Ind', 'Pell_Grant', 'Fee_Assistance_Program',
            'Childhood_Med_Underserved_Self_Reported',
            
            # Other binary flags
            'Prev_Applied_Rush', 'Inst_Action_Ind',
            'Investigation_Ind', 'Felony_Ind', 'Misdemeanor_Ind',
            'Military_Service',
            
            # Numeric features
            'Number_in_Household', 'Num_Dependents',
            'Service Rating (Numerical)'
        ]
        
        # LLM features
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
        print("SIMPLIFIED INTEGRATED MEDICAL ADMISSIONS MODEL")
        print("="*80)
        
    def load_and_prepare_data(self, year: int, llm_scores_file: str) -> pd.DataFrame:
        """Load structured data and merge with LLM scores"""
        
        # Load structured data
        year_path = self.data_base_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        df = pd.read_excel(year_path / "1. Applicants.xlsx")
        
        # Standardize ID column
        if 'Amcas_ID' in df.columns:
            df['AMCAS_ID'] = df['Amcas_ID'].astype(str)
        elif 'amcas_id' in df.columns:
            df['AMCAS_ID'] = df['amcas_id'].astype(str)
            
        # Load LLM scores
        llm_df = pd.read_csv(llm_scores_file)
        llm_df['AMCAS_ID'] = llm_df['AMCAS_ID_original'].astype(str)
        
        # Filter LLM scores for this year
        if 'year' in llm_df.columns:
            llm_df = llm_df[llm_df['year'] == year].copy()
        
        # Merge on AMCAS_ID
        df = df.merge(
            llm_df[['AMCAS_ID'] + self.llm_features],
            on='AMCAS_ID',
            how='left'
        )
        
        df['year'] = year
        
        # Create derived features
        epsilon = 1e-8
        if 'Exp_Hour_Total' in df.columns and df['Exp_Hour_Total'].notna().any():
            df['research_proportion'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
            df['clinical_proportion'] = (df['Exp_Hour_Employ_Med'] + df['Exp_Hour_Shadowing']) / (df['Exp_Hour_Total'] + epsilon)
            df['volunteer_proportion'] = (df['Exp_Hour_Volunteer_Med'] + df['Exp_Hour_Volunteer_Non_Med']) / (df['Exp_Hour_Total'] + epsilon)
        
        # GPA difference
        if 'Total_GPA' in df.columns and 'BCPM_GPA' in df.columns:
            df['gpa_difference'] = df['Total_GPA'] - df['BCPM_GPA']
        
        return df
        
    def prepare_features(self, df: pd.DataFrame, require_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """Extract features and target"""
        
        # Get available features
        available_structured = [f for f in self.structured_features if f in df.columns]
        available_llm = [f for f in self.llm_features if f in df.columns]
        derived_features = ['research_proportion', 'clinical_proportion', 'volunteer_proportion', 'gpa_difference']
        available_derived = [f for f in derived_features if f in df.columns]
        
        all_features = available_structured + available_llm + available_derived
        
        # Extract features
        X = df[all_features].copy()
        
        # Convert binary columns (Yes/No to 1/0)
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if it's a Yes/No column
                unique_vals = X[col].dropna().unique()
                if set(unique_vals).issubset({'Yes', 'No', 'Y', 'N'}):
                    X[col] = X[col].map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0})
                # Check if it's Male/Female
                elif set(unique_vals).issubset({'Male', 'Female', 'M', 'F'}):
                    X[col] = X[col].map({'Male': 1, 'M': 1, 'Female': 0, 'F': 0})
                # Otherwise drop it
                else:
                    print(f"  Warning: Dropping non-numeric column '{col}' with values: {unique_vals[:5]}")
                    X = X.drop(columns=[col])
                    all_features.remove(col)
        
        # Extract target if available
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
        elif require_target:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
        else:
            y = None
        
        print(f"\n✓ Features prepared:")
        print(f"  - Structured: {len([f for f in available_structured if f in all_features])}")
        print(f"  - LLM: {len([f for f in available_llm if f in all_features])}")
        print(f"  - Derived: {len([f for f in available_derived if f in all_features])}")
        print(f"  - Total: {len(all_features)}")
        if y is not None:
            print(f"  - Target: Available (mean={y.mean():.1f})")
        else:
            print(f"  - Target: Not available (prediction mode)")
        
        return X, y, all_features
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model with 5-fold cross-validation"""
        print("\n[STEP 3] Training Model")
        print("-" * 50)
        
        # Impute missing values
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
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
                n_jobs=-1,
                objective='reg:squarederror'
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
            
        # 5-fold cross-validation
        print("\n✓ Performing 5-fold cross-validation...")
        
        # Create stratification bins
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
        
        # Train final model
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
            
        # Separate LLM vs structured
        llm_importance = sum(importances[i] for i, name in enumerate(feature_names) 
                           if i < len(importances) and 'llm_' in name)
        total_importance = sum(importances)
        
        if total_importance > 0:
            print(f"\n✓ Feature Group Importance:")
            print(f"  LLM features:        {llm_importance/total_importance:.1%}")
            print(f"  Structured features: {(total_importance-llm_importance)/total_importance:.1%}")
        
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance on holdout test set"""
        print("\n[STEP 4] Evaluating on 2024 Holdout Test Set")
        print("-" * 50)
        
        # Preprocess test data
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
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
        
        # Classification metrics
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
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
    def save_model(self, metrics: Dict, feature_names: List[str]) -> str:
        """Save trained model and components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': feature_names,
            'llm_features': self.llm_features,
            'structured_features': self.structured_features,
            'target_column': self.target_column,
            'interview_threshold': self.interview_threshold,
            'training_metrics': metrics,
            'timestamp': timestamp
        }
        
        model_path = self.models_path / f"integrated_model_{timestamp}.pkl"
        joblib.dump(model_data, model_path)
        
        # Also save as 'latest'
        latest_path = self.models_path / "integrated_model_latest.pkl"
        joblib.dump(model_data, latest_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Also saved as: {latest_path}")
        
        return str(model_path)
        
    def run_pipeline(self) -> str:
        """Execute the complete training pipeline"""
        
        # Step 1: Load training data
        print("\n[STEP 1] Loading Training Data (2022-2023)")
        print("-" * 50)
        
        df_2022 = self.load_and_prepare_data(2022, "llm_scores_2022_2023_20250619_172837.csv")
        print(f"✓ Loaded 2022: {len(df_2022)} applicants")
        
        df_2023 = self.load_and_prepare_data(2023, "llm_scores_2022_2023_20250619_172837.csv")
        print(f"✓ Loaded 2023: {len(df_2023)} applicants")
        
        train_df = pd.concat([df_2022, df_2023], ignore_index=True)
        print(f"✓ Combined training data: {len(train_df)} total applicants")
        
        # Check target
        if self.target_column in train_df.columns:
            print(f"\n✓ Target variable '{self.target_column}' found")
            print(f"  Score distribution: mean={train_df[self.target_column].mean():.1f}, std={train_df[self.target_column].std():.1f}")
            print(f"  Interview rate (≥19): {(train_df[self.target_column] >= self.interview_threshold).mean():.1%}")
        
        # Step 2: Load test data
        print("\n[STEP 2] Loading Test Data (2024)")
        print("-" * 50)
        
        # Find latest 2024 LLM scores
        try:
            with open("latest_2024_llm_scores.txt", "r") as f:
                llm_file = f.read().strip()
        except:
            llm_file = "llm_scores_2024_20250619_175241.csv"
            
        test_df = self.load_and_prepare_data(2024, llm_file)
        print(f"✓ Loaded 2024 test data: {len(test_df)} applicants")
        
        # Prepare features
        X_train, y_train, train_features = self.prepare_features(train_df, require_target=True)
        X_test, y_test, test_features = self.prepare_features(test_df, require_target=False)
        
        # Use only common features
        common_features = [f for f in train_features if f in test_features]
        print(f"\n✓ Using {len(common_features)} common features")
        
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        feature_names = common_features
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate or predict
        if y_test is not None:
            # We have ground truth - evaluate
            metrics = self.evaluate_on_test_set(X_test, y_test)
        else:
            # No ground truth - just make predictions
            print("\n[STEP 4] Making Predictions on 2024 Data")
            print("-" * 50)
            print("Note: No ground truth available for 2024 data")
            
            # Preprocess test data
            X_test_imputed = self.imputer.transform(X_test)
            X_test_scaled = self.scaler.transform(X_test_imputed)
            
            # Predict scores
            y_pred = self.model.predict(X_test_scaled)
            
            print(f"\n✓ Predictions made for {len(y_pred)} applicants")
            print(f"  Predicted score distribution:")
            print(f"    Mean:  {np.mean(y_pred):.1f}")
            print(f"    Std:   {np.std(y_pred):.1f}")
            print(f"    Min:   {np.min(y_pred):.1f}")
            print(f"    Max:   {np.max(y_pred):.1f}")
            print(f"  Predicted interview rate (≥19): {(y_pred >= self.interview_threshold).mean():.1%}")
            
            # Save predictions
            predictions_df = test_df[['AMCAS_ID']].copy()
            predictions_df['predicted_score'] = y_pred
            predictions_df['predicted_interview'] = (y_pred >= self.interview_threshold).astype(int)
            predictions_file = f"predictions_2024_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(predictions_file, index=False)
            print(f"\n✓ Predictions saved to: {predictions_file}")
            
            metrics = {
                'test_predictions': y_pred,
                'test_size': len(y_pred),
                'predicted_interview_rate': (y_pred >= self.interview_threshold).mean()
            }
        
        # Save model
        model_path = self.save_model(metrics, feature_names)
        
        print("\n" + "="*80)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*80)
        
        return model_path


def main():
    """Execute the simplified integrated training pipeline"""
    pipeline = SimpleIntegratedPipeline()
    model_path = pipeline.run_pipeline()
    return model_path


if __name__ == "__main__":
    main()