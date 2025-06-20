"""
Train Ordinal Regression Model with Enhanced Features
===================================================

Complete pipeline for training the 4-bucket ordinal classifier
using enhanced features and natural break boundaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ordinal_regression_model import OrdinalXGBoostClassifier, evaluate_ordinal_model
from enhanced_features import EnhancedFeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class OrdinalTrainingPipeline:
    """
    Complete training pipeline for ordinal regression model.
    """
    
    def __init__(self):
        self.data_path = Path("data")
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Natural break boundaries for buckets
        self.bucket_boundaries = [0, 10, 16, 22, 26]
        self.bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
        
        # Components
        self.feature_engineer = EnhancedFeatureEngineer(self.bucket_boundaries)
        self.ordinal_model = OrdinalXGBoostClassifier(
            n_classes=4,
            bucket_boundaries=self.bucket_boundaries,
            n_estimators=800,
            max_depth=8,
            learning_rate=0.015
        )
        
        # Data containers
        self.train_data = None
        self.test_data = None
        
    def load_training_data(self) -> pd.DataFrame:
        """Load and combine 2022-2023 training data with LLM scores."""
        print("\n[STEP 1] Loading Training Data")
        print("-" * 50)
        
        all_data = []
        
        for year in [2022, 2023]:
            # Load structured data
            year_path = self.data_path / f"{year} Applicants Reviewed by Trusted Reviewers"
            df = pd.read_excel(year_path / "1. Applicants.xlsx")
            
            # Standardize ID
            if 'Amcas_ID' in df.columns:
                df['AMCAS_ID'] = df['Amcas_ID'].astype(str)
            
            df['year'] = year
            all_data.append(df)
        
        # Combine years
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Load LLM scores
        llm_df = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
        llm_df['AMCAS_ID'] = llm_df['AMCAS_ID_original'].astype(str)
        
        # Merge with LLM scores
        llm_cols = [col for col in llm_df.columns if col.startswith('llm_')]
        combined_df = combined_df.merge(
            llm_df[['AMCAS_ID'] + llm_cols],
            on='AMCAS_ID',
            how='left'
        )
        
        # Fill missing LLM scores with median
        for col in llm_cols:
            if col in combined_df.columns:
                combined_df[col].fillna(combined_df[col].median(), inplace=True)
        
        print(f"✓ Loaded {len(combined_df)} training samples")
        print(f"✓ Features: {len(combined_df.columns)} columns")
        
        # Verify target exists
        if 'Application Review Score' not in combined_df.columns:
            raise ValueError("Target column 'Application Review Score' not found!")
        
        # Show score distribution
        scores = combined_df['Application Review Score']
        print(f"\n✓ Score Distribution:")
        print(f"  Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")
        print(f"  Range: [{scores.min()}, {scores.max()}]")
        
        return combined_df
    
    def load_test_data(self) -> pd.DataFrame:
        """Load 2024 test data with LLM scores."""
        print("\n[STEP 2] Loading Test Data (2024)")
        print("-" * 50)
        
        # Load structured data
        year_path = self.data_path / "2024 Applicants Reviewed by Trusted Reviewers"
        df = pd.read_excel(year_path / "1. Applicants.xlsx")
        
        # Standardize ID
        if 'Amcas_ID' in df.columns:
            df['AMCAS_ID'] = df['Amcas_ID'].astype(str)
        
        # Load LLM scores
        try:
            with open("latest_2024_llm_scores.txt", "r") as f:
                llm_file = f.read().strip()
        except:
            llm_file = "llm_scores_2024_20250619_175241.csv"
        
        llm_df = pd.read_csv(llm_file)
        llm_df['AMCAS_ID'] = llm_df['AMCAS_ID_original'].astype(str)
        
        # Merge
        llm_cols = [col for col in llm_df.columns if col.startswith('llm_')]
        df = df.merge(
            llm_df[['AMCAS_ID'] + llm_cols],
            on='AMCAS_ID',
            how='left'
        )
        
        # Fill missing
        for col in llm_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        print(f"✓ Loaded {len(df)} test samples")
        
        # Check if we have ground truth
        has_target = 'Application Review Score' in df.columns
        print(f"✓ Ground truth available: {has_target}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Prepare features and target."""
        
        # Extract target if available
        target = None
        if 'Application Review Score' in df.columns:
            scores = df['Application Review Score'].values
            # Convert to buckets
            target = np.digitize(scores, self.bucket_boundaries[1:-1])
            
            # Show bucket distribution
            print(f"\n✓ Bucket Distribution:")
            for i, name in enumerate(self.bucket_names):
                count = np.sum(target == i)
                pct = count / len(target) * 100
                print(f"  {name}: {count} ({pct:.1f}%)")
        
        # Engineer features
        if fit:
            X = self.feature_engineer.fit_transform(df)
        else:
            X = self.feature_engineer.transform(df)
        
        return X, target
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                   X_val: pd.DataFrame, y_val: np.ndarray) -> Dict:
        """Train ordinal regression model."""
        print("\n[STEP 3] Training Ordinal Model")
        print("-" * 50)
        
        # Train model
        self.ordinal_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True
        )
        
        # Evaluate on validation set
        print("\n✓ Validation Set Performance:")
        val_results = evaluate_ordinal_model(
            self.ordinal_model, X_val, y_val, self.bucket_names
        )
        
        return val_results
    
    def run_training_pipeline(self) -> str:
        """Execute complete training pipeline."""
        print("="*80)
        print("ORDINAL REGRESSION TRAINING PIPELINE")
        print("="*80)
        
        # Load data
        train_df = self.load_training_data()
        test_df = self.load_test_data()
        
        # Prepare training data
        X_train_full, y_train_full = self.prepare_data(train_df, fit=True)
        
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, 
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\n✓ Training split: {len(X_train)} samples")
        print(f"✓ Validation split: {len(X_val)} samples")
        
        # Train model
        val_results = self.train_model(X_train, y_train, X_val, y_val)
        
        # Prepare test data
        X_test, y_test = self.prepare_data(test_df, fit=False)
        
        if y_test is not None:
            # Evaluate on test set
            print("\n[STEP 4] Test Set Evaluation")
            print("-" * 50)
            test_results = evaluate_ordinal_model(
                self.ordinal_model, X_test, y_test, self.bucket_names
            )
        else:
            # Make predictions
            print("\n[STEP 4] Making Predictions on 2024 Data")
            print("-" * 50)
            
            predictions, confidences = self.ordinal_model.predict_with_confidence(X_test)
            
            print(f"\n✓ Predictions for {len(predictions)} applicants")
            print(f"\n✓ Predicted Distribution:")
            for i, name in enumerate(self.bucket_names):
                count = np.sum(predictions == i)
                pct = count / len(predictions) * 100
                print(f"  {name}: {count} ({pct:.1f}%)")
            
            print(f"\n✓ Average Confidence: {np.mean(confidences):.1%}")
            
            # Save predictions
            pred_df = pd.DataFrame({
                'AMCAS_ID': test_df['AMCAS_ID'],
                'predicted_bucket': predictions,
                'bucket_name': [self.bucket_names[p] for p in predictions],
                'confidence': confidences
            })
            
            # Add probability for each bucket
            proba = self.ordinal_model.predict_proba(X_test)
            for i, name in enumerate(self.bucket_names):
                pred_df[f'prob_{name}'] = proba[:, i]
            
            pred_file = f"ordinal_predictions_2024_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            pred_df.to_csv(pred_file, index=False)
            print(f"\n✓ Predictions saved to: {pred_file}")
            
            test_results = None
        
        # Save model
        print("\n[STEP 5] Saving Model")
        print("-" * 50)
        
        model_data = {
            'ordinal_model': self.ordinal_model,
            'feature_engineer': self.feature_engineer,
            'bucket_boundaries': self.bucket_boundaries,
            'bucket_names': self.bucket_names,
            'validation_results': val_results,
            'test_results': test_results,
            'feature_importance': self.ordinal_model.get_feature_importance(),
            'timestamp': datetime.now()
        }
        
        model_path = self.models_path / f"ordinal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model_data, model_path)
        
        # Also save as latest
        latest_path = self.models_path / "ordinal_model_latest.pkl"
        joblib.dump(model_data, latest_path)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Also saved as: {latest_path}")
        
        # Print top features
        print("\n✓ Top 15 Important Features:")
        importance = self.ordinal_model.get_feature_importance()
        for i, (feature, score) in enumerate(list(importance.items())[:15]):
            print(f"  {i+1:2d}. {feature:40s} {score:.4f}")
        
        # Feature group importance
        feature_groups = self.feature_engineer.get_feature_groups()
        group_importance = {}
        for group, features in feature_groups.items():
            group_importance[group] = sum(importance.get(f, 0) for f in features)
        
        print("\n✓ Feature Group Importance:")
        total_imp = sum(group_importance.values())
        for group, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
            if imp > 0:
                print(f"  {group}: {imp/total_imp:.1%}")
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        
        return str(model_path)


def main():
    """Execute the training pipeline."""
    pipeline = OrdinalTrainingPipeline()
    model_path = pipeline.run_training_pipeline()
    return model_path


if __name__ == "__main__":
    main()