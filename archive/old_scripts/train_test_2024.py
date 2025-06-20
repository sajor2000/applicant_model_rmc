"""
Train on 2022-2023 Combined Data, Test on 2024 Holdout Set
===========================================================

This script explicitly:
1. Loads and combines 2022 and 2023 data for training
2. Uses 2024 data as a completely separate holdout test set
3. Ensures consistent feature engineering across all years
4. Reports performance metrics specifically on 2024 test data

Author: Medical Admissions AI Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, cohen_kappa_score
)

# Import existing models for comparison
from four_tier_classifier import HighConfidenceFourTierClassifier
from model_comparison import (
    OrdinalClassifier, CostSensitiveRF, create_cost_matrix
)

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")


class Train2024TestPipeline:
    """
    Main pipeline for training on 2022-2023 and testing on 2024 data
    """
    
    def __init__(self):
        self.data_base_path = Path("data")
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Data containers
        self.train_data = None  # Combined 2022-2023
        self.test_data = None   # 2024 only
        self.feature_names = None
        self.scaler = StandardScaler()
        
        print("="*80)
        print("MEDICAL ADMISSIONS MODEL TRAINING PIPELINE")
        print("Training Data: 2022 + 2023 Combined")
        print("Test Data: 2024 (Holdout Set)")
        print("="*80)
        
    def load_and_combine_training_data(self) -> pd.DataFrame:
        """Load 2022 and 2023 data and combine for training"""
        print("\n[STEP 1] Loading Training Data (2022 + 2023)")
        print("-" * 50)
        
        # Load 2022 data
        path_2022 = self.data_base_path / "2022 Applicants Reviewed by Trusted Reviewers" / "1. Applicants.xlsx"
        df_2022 = pd.read_excel(path_2022)
        df_2022['year'] = 2022
        print(f"✓ Loaded 2022 data: {len(df_2022)} applicants")
        
        # Load 2023 data
        path_2023 = self.data_base_path / "2023 Applicants Reviewed by Trusted Reviewers" / "1. Applicants.xlsx"
        df_2023 = pd.read_excel(path_2023)
        df_2023['year'] = 2023
        print(f"✓ Loaded 2023 data: {len(df_2023)} applicants")
        
        # Combine training data
        self.train_data = pd.concat([df_2022, df_2023], ignore_index=True)
        print(f"\n✓ Combined training data: {len(self.train_data)} total applicants")
        
        return self.train_data
    
    def load_test_data(self) -> pd.DataFrame:
        """Load 2024 data as holdout test set"""
        print("\n[STEP 2] Loading Test Data (2024 Holdout)")
        print("-" * 50)
        
        path_2024 = self.data_base_path / "2024 Applicants Reviewed by Trusted Reviewers" / "1. Applicants.xlsx"
        self.test_data = pd.read_excel(path_2024)
        self.test_data['year'] = 2024
        print(f"✓ Loaded 2024 test data: {len(self.test_data)} applicants")
        
        return self.test_data
    
    def preprocess_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply consistent preprocessing to any dataset
        
        Args:
            df: Input dataframe
            fit_scaler: Whether to fit the scaler (True for training data)
        
        Returns:
            Processed features and target array
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        numeric_cols = [
            'Age', 'Exp_Hour_Total', 'Exp_Hour_Research', 
            'Exp_Hour_Volunteer_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Service Rating (Numerical)', 'Num_Dependents'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle GPA trends
        gpa_cols = ['Total_GPA_Trend', 'BCPM_GPA_Trend']
        for col in gpa_cols:
            if col in df.columns:
                df[col] = df[col].replace('NULL', 0).fillna(0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Feature engineering
        epsilon = 1e-6
        df['research_intensity'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
        df['clinical_intensity'] = (
            (df.get('Exp_Hour_Volunteer_Med', 0) + df.get('Exp_Hour_Shadowing', 0)) / 
            (df['Exp_Hour_Total'] + epsilon)
        )
        df['experience_balance'] = (
            df['Exp_Hour_Research'] / 
            (df.get('Exp_Hour_Volunteer_Med', 0) + df.get('Exp_Hour_Shadowing', 0) + epsilon)
        )
        df['service_commitment'] = (
            df.get('Service Rating (Numerical)', 0) * 
            np.log(df['Comm_Service_Total_Hours'] + 1)
        )
        
        # Handle categorical features
        binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).map({'Yes': 1, 'yes': 1}).fillna(0).astype(int)
        
        # Create adversity_overcome feature
        if 'Disadvantanged_Ind' in df.columns:
            df['adversity_overcome'] = df['Disadvantanged_Ind'] * df.get('Total_GPA_Trend', 0)
        
        # Define target
        if 'Application Review Score' in df.columns:
            def assign_tier(score):
                if pd.isna(score): return None
                score = pd.to_numeric(score, errors='coerce')
                if pd.isna(score): return None
                if score <= 14: return 0  # Very Unlikely
                if 15 <= score <= 18: return 1  # Potential Review
                if 19 <= score <= 22: return 2  # Probable Interview
                return 3  # Very Likely Interview
            
            df['target'] = df['Application Review Score'].apply(assign_tier)
            df = df.dropna(subset=['target'])
            target = df['target'].values.astype(int)
        else:
            raise ValueError("No 'Application Review Score' column found!")
        
        # Select features
        feature_cols = [
            'Age', 'Exp_Hour_Total', 'Exp_Hour_Research',
            'Exp_Hour_Volunteer_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Service Rating (Numerical)', 'Total_GPA_Trend',
            'BCPM_GPA_Trend', 'research_intensity', 'clinical_intensity',
            'experience_balance', 'service_commitment', 'adversity_overcome',
            'Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value'
        ]
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        features = df[feature_cols].copy()
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = feature_cols
        
        # Scale features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        return pd.DataFrame(features_scaled, columns=feature_cols), target
    
    def train_models(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict:
        """Train multiple models on the training data"""
        print("\n[STEP 4] Training Models on 2022-2023 Combined Data")
        print("-" * 50)
        
        models = {}
        
        # 1. Random Forest (Baseline)
        print("\n1. Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        print("✓ Random Forest trained")
        
        # 2. Cost-Sensitive Random Forest
        print("\n2. Training Cost-Sensitive Random Forest...")
        cost_rf = CostSensitiveRF(
            cost_matrix=create_cost_matrix(),
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        cost_rf.fit(X_train, y_train)
        models['CostSensitiveRF'] = cost_rf
        print("✓ Cost-Sensitive RF trained")
        
        # 3. Ordinal Classifier
        print("\n3. Training Ordinal Classifier...")
        ordinal_model = OrdinalClassifier()
        ordinal_model.fit(X_train, y_train)
        models['OrdinalClassifier'] = ordinal_model
        print("✓ Ordinal Classifier trained")
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n4. Training XGBoost Classifier...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=4,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            print("✓ XGBoost trained")
        
        return models
    
    def evaluate_on_2024(self, models: Dict, X_test: pd.DataFrame, y_test: np.ndarray):
        """Evaluate all models specifically on 2024 holdout data"""
        print("\n[STEP 5] Evaluating Models on 2024 Test Data")
        print("="*80)
        
        results = {}
        tier_names = ['Very Unlikely', 'Potential Review', 'Probable Interview', 'Very Likely Interview']
        
        for model_name, model in models.items():
            print(f"\n{model_name} Performance on 2024 Data:")
            print("-" * 50)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_per_class = f1_score(y_test, y_pred, average=None)
            kappa = cohen_kappa_score(y_test, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_per_class': f1_per_class,
                'kappa': kappa,
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            # Print metrics
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Weighted F1: {f1_weighted:.3f}")
            print(f"Cohen's Kappa: {kappa:.3f}")
            print(f"\nF1 Score per Tier:")
            for i, (tier, f1) in enumerate(zip(tier_names, f1_per_class)):
                print(f"  {tier}: {f1:.3f}")
            
            print(f"\nConfusion Matrix:")
            print(cm)
            
            # Classification report
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, target_names=tier_names))
        
        return results
    
    def save_best_model(self, models: Dict, results: Dict):
        """Save the best performing model"""
        print("\n[STEP 6] Selecting and Saving Best Model")
        print("-" * 50)
        
        # Find best model based on weighted F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_model = models[best_model_name]
        best_score = results[best_model_name]['f1_weighted']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Weighted F1 Score on 2024 Test Data: {best_score:.3f}")
        
        # Save model with metadata
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_years': [2022, 2023],
            'test_year': 2024,
            'test_performance': results[best_model_name],
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = self.models_path / f"best_model_2022_2023_train_2024_test.pkl"
        joblib.dump(model_data, save_path)
        print(f"\n✓ Model saved to: {save_path}")
        
        # Also save all results for comparison
        all_results_path = self.models_path / "all_results_2024_test.pkl"
        joblib.dump(results, all_results_path)
        print(f"✓ All results saved to: {all_results_path}")
        
        return best_model_name, best_model
    
    def run_pipeline(self):
        """Execute the complete training and evaluation pipeline"""
        print("\nStarting Pipeline Execution...")
        print("="*80)
        
        # Load data
        train_df = self.load_and_combine_training_data()
        test_df = self.load_test_data()
        
        # Preprocess data
        print("\n[STEP 3] Preprocessing Data")
        print("-" * 50)
        X_train, y_train = self.preprocess_data(train_df, fit_scaler=True)
        print(f"✓ Training data processed: {X_train.shape}")
        
        X_test, y_test = self.preprocess_data(test_df, fit_scaler=False)
        print(f"✓ Test data processed: {X_test.shape}")
        
        # Train models
        models = self.train_models(X_train, y_train)
        
        # Evaluate on 2024
        results = self.evaluate_on_2024(models, X_test, y_test)
        
        # Save best model
        best_model_name, best_model = self.save_best_model(models, results)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print(f"Best Model: {best_model_name}")
        print(f"Test Performance (2024): {results[best_model_name]['f1_weighted']:.3f} weighted F1")
        print("="*80)
        
        return models, results


def main():
    """Main execution function"""
    pipeline = Train2024TestPipeline()
    models, results = pipeline.run_pipeline()
    
    # Print summary
    print("\n\nSUMMARY OF ALL MODELS ON 2024 TEST DATA:")
    print("-"*50)
    for model_name, result in results.items():
        print(f"{model_name:20s} - Accuracy: {result['accuracy']:.3f}, F1: {result['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()