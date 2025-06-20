"""
Train Model with Filtered Consistent Data
========================================

This script trains the ordinal regression model using only the filtered,
consistent features that have <75% missing data across all years.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
from ordinal_regression_model import OrdinalXGBoostClassifier
from enhanced_features import EnhancedFeatureEngineer


class FilteredDataTrainingPipeline:
    """Training pipeline using filtered, consistent data."""
    
    def __init__(self):
        self.filtered_data_path = Path("data_filtered")
        self.llm_scores_path = Path(".")
        self.model_save_path = Path("models")
        self.model_save_path.mkdir(exist_ok=True)
        
        # Load consistent features list
        import json
        with open("consistent_features_list.json", "r") as f:
            self.feature_info = json.load(f)
        
        self.consistent_features = self.feature_info['consistent_features']
        
    def load_filtered_data(self) -> pd.DataFrame:
        """Load filtered training data (2022-2023)."""
        print("\n1. Loading filtered training data...")
        
        # Load 2022 and 2023 data
        df_2022 = pd.read_excel(self.filtered_data_path / "2022_filtered_applicants.xlsx")
        df_2023 = pd.read_excel(self.filtered_data_path / "2023_filtered_applicants.xlsx")
        
        # Combine
        df_train = pd.concat([df_2022, df_2023], ignore_index=True)
        print(f"   Loaded {len(df_2022)} from 2022, {len(df_2023)} from 2023")
        print(f"   Total training samples: {len(df_train)}")
        
        return df_train
    
    def load_llm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge LLM scores."""
        print("\n2. Loading LLM scores...")
        
        # Try to load the most recent LLM scores
        llm_files = list(Path(".").glob("llm_scores_2022_2023_*.csv"))
        if llm_files:
            latest_llm = sorted(llm_files)[-1]
            df_llm = pd.read_csv(latest_llm)
            print(f"   Loaded LLM scores from: {latest_llm}")
            print(f"   LLM columns: {list(df_llm.columns)[:10]}")
            
            # Check for AMCAS ID column (might be uppercase or different name)
            if 'amcas_id' not in df_llm.columns:
                if 'AMCAS_ID' in df_llm.columns:
                    df_llm = df_llm.rename(columns={'AMCAS_ID': 'amcas_id'})
                elif 'AMCAS_ID_original' in df_llm.columns:
                    df_llm = df_llm.rename(columns={'AMCAS_ID_original': 'amcas_id'})
            
            # Get available LLM columns
            llm_cols = [col for col in df_llm.columns if col.startswith('llm_')]
            if llm_cols:
                # Add amcas_id to merge columns
                merge_cols = ['amcas_id'] + llm_cols[:10]  # Limit to first 10 LLM features
                available_cols = [col for col in merge_cols if col in df_llm.columns]
                
                # Merge
                df = df.merge(
                    df_llm[available_cols],
                    on='amcas_id',
                    how='left'
                )
                print(f"   Merged {len(available_cols)-1} LLM features for {df[llm_cols[0]].notna().sum()} applicants")
            else:
                print("   No LLM columns found with 'llm_' prefix")
        else:
            print("   No LLM scores found - proceeding with structured data only")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        print("\n3. Preparing features...")
        
        # Get target variable
        y = df['application_review_score'].values
        
        # Define feature columns (exclude target and ID)
        exclude_cols = ['application_review_score', 'amcas_id', 'appl_year']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove duplicates
        feature_cols = list(set(feature_cols))
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"   Using {len(available_features)} features")
        
        # Separate numeric and categorical
        numeric_features = []
        categorical_features = []
        
        for col in available_features:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f"   - Numeric features: {len(numeric_features)}")
        print(f"   - Categorical features: {len(categorical_features)}")
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle categorical features
        for col in categorical_features:
            if col == 'gender':
                X[col] = X[col].map({'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1})
            elif col == 'citizenship':
                X[col] = X[col].map({'US_Citizen': 0, 'Permanent_Resident': 1, 'International': 2})
            elif col == 'service_rating_categorical':
                # Map service rating categories to numeric
                rating_map = {
                    'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                    'Good': 2, 'Average': 1, 'Below Average': 0,
                    'Poor': -1, np.nan: -2
                }
                X[col] = X[col].map(rating_map).fillna(-2)
            else:
                # For other categorical, use simple encoding
                X[col] = pd.Categorical(X[col]).codes
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Create DataFrame with scaled features
        X_df = pd.DataFrame(X_scaled, columns=available_features)
        
        return X_df, y, available_features, imputer, scaler
    
    def train_ordinal_model(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """Train ordinal regression model with cross-validation."""
        print("\n4. Training ordinal regression model...")
        
        # Define bucket boundaries based on natural breaks
        bucket_boundaries = [0, 10, 16, 22, 26]
        
        # Initialize model
        model = OrdinalXGBoostClassifier(
            n_classes=4,
            bucket_boundaries=bucket_boundaries,
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Convert continuous scores to buckets for stratification
        y_buckets = np.digitize(y, bucket_boundaries[1:-1])
        
        cv_scores = []
        cv_adjacent = []
        cv_qwk = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_buckets)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            cv_scores.append(metrics['exact_match_accuracy'])
            cv_adjacent.append(metrics['adjacent_accuracy'])
            cv_qwk.append(metrics['quadratic_weighted_kappa'])
            
            print(f"   Fold {fold+1}: Exact={metrics['exact_match_accuracy']:.3f}, "
                  f"Adjacent={metrics['adjacent_accuracy']:.3f}, "
                  f"QWK={metrics['quadratic_weighted_kappa']:.3f}")
        
        print(f"\n   Average CV Performance:")
        print(f"   - Exact Match: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
        print(f"   - Adjacent: {np.mean(cv_adjacent):.3f} (±{np.std(cv_adjacent):.3f})")
        print(f"   - QWK: {np.mean(cv_qwk):.3f} (±{np.std(cv_qwk):.3f})")
        
        # Train final model on all data
        print("\n5. Training final model on all data...")
        model.fit(X, y)
        
        return model, {
            'cv_exact_match': np.mean(cv_scores),
            'cv_adjacent': np.mean(cv_adjacent),
            'cv_qwk': np.mean(cv_qwk)
        }
    
    def evaluate_on_2024(self, model, feature_cols: list, 
                        imputer, scaler) -> pd.DataFrame:
        """Evaluate model on 2024 holdout data."""
        print("\n6. Evaluating on 2024 holdout data...")
        
        # Load 2024 data
        df_2024 = pd.read_excel(self.filtered_data_path / "2024_filtered_applicants.xlsx")
        print(f"   Loaded {len(df_2024)} test samples from 2024")
        
        # Load LLM scores for 2024
        df_2024 = self.load_llm_scores(df_2024)
        
        # Prepare features
        X_test = df_2024[feature_cols].copy()
        
        # Handle categorical features (same as training)
        categorical_features = X_test.select_dtypes(exclude=['int64', 'float64']).columns
        
        for col in categorical_features:
            if col == 'gender':
                X_test[col] = X_test[col].map({'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1})
            elif col == 'citizenship':
                X_test[col] = X_test[col].map({'US_Citizen': 0, 'Permanent_Resident': 1, 'International': 2})
            elif col == 'service_rating_categorical':
                rating_map = {
                    'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                    'Good': 2, 'Average': 1, 'Below Average': 0,
                    'Poor': -1, np.nan: -2
                }
                X_test[col] = X_test[col].map(rating_map).fillna(-2)
            else:
                X_test[col] = pd.Categorical(X_test[col]).codes
        
        # Apply same preprocessing
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        # Make predictions
        predictions, confidence = model.predict_with_confidence(X_test_df)
        
        # Get bucket labels
        bucket_labels = ['Reject', 'Waitlist', 'Interview', 'Accept']
        
        # Create results DataFrame
        results = pd.DataFrame({
            'amcas_id': df_2024['amcas_id'],
            'true_score': df_2024['application_review_score'],
            'true_bucket': pd.cut(
                df_2024['application_review_score'],
                bins=[0, 10, 16, 22, 26],
                labels=bucket_labels,
                include_lowest=True
            ),
            'predicted_bucket_idx': predictions,
            'predicted_bucket': [bucket_labels[i] for i in predictions],
            'confidence': confidence,
            'service_rating': df_2024['service_rating_numerical']
        })
        
        # Calculate metrics
        y_true = df_2024['application_review_score'].values
        metrics = model.evaluate(X_test_df, y_true)
        
        print(f"\n   2024 Test Set Performance:")
        print(f"   - Exact Match: {metrics['exact_match_accuracy']:.3f}")
        print(f"   - Adjacent: {metrics['adjacent_accuracy']:.3f}")
        print(f"   - QWK: {metrics['quadratic_weighted_kappa']:.3f}")
        
        # Analyze predictions
        pred_dist = results['predicted_bucket'].value_counts()
        print(f"\n   Prediction Distribution:")
        for bucket in bucket_labels:
            count = pred_dist.get(bucket, 0)
            pct = count / len(results) * 100
            print(f"   - {bucket}: {count} ({pct:.1f}%)")
        
        return results, metrics
    
    def save_model(self, model, feature_cols: list, imputer, scaler, 
                  cv_metrics: dict, test_metrics: dict):
        """Save trained model and preprocessing objects."""
        print("\n7. Saving model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_save_path / f"ordinal_model_filtered_{timestamp}.pkl"
        
        model_data = {
            'model': model,
            'feature_columns': feature_cols,
            'imputer': imputer,
            'scaler': scaler,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'training_date': timestamp,
            'n_features': len(feature_cols),
            'consistent_features': self.consistent_features
        }
        
        joblib.dump(model_data, model_path)
        print(f"   Model saved to: {model_path}")
        
        # Also save as latest
        latest_path = self.model_save_path / "ordinal_model_filtered_latest.pkl"
        joblib.dump(model_data, latest_path)
        print(f"   Also saved as: {latest_path}")
        
        return model_path
    
    def run_training_pipeline(self):
        """Run complete training pipeline."""
        print("="*80)
        print("TRAINING ORDINAL MODEL WITH FILTERED CONSISTENT DATA")
        print("="*80)
        
        # Load data
        df_train = self.load_filtered_data()
        df_train = self.load_llm_scores(df_train)
        
        # Prepare features
        X, y, feature_cols, imputer, scaler = self.prepare_features(df_train)
        
        # Train model
        model, cv_metrics = self.train_ordinal_model(X, y)
        
        # Evaluate on 2024
        results_2024, test_metrics = self.evaluate_on_2024(
            model, feature_cols, imputer, scaler
        )
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"predictions_2024_filtered_{timestamp}.csv"
        results_2024.to_csv(results_path, index=False)
        print(f"\n   Predictions saved to: {results_path}")
        
        # Save model
        model_path = self.save_model(
            model, feature_cols, imputer, scaler, cv_metrics, test_metrics
        )
        
        # Feature importance
        print("\n8. Top Feature Importance:")
        importance_df = model.get_feature_importance(feature_cols)
        for _, row in importance_df.head(15).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.3f}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print(f"Model performance on 2024 test set:")
        print(f"- Exact Match: {test_metrics['exact_match_accuracy']:.3f}")
        print(f"- Adjacent Accuracy: {test_metrics['adjacent_accuracy']:.3f}")
        print(f"- QWK: {test_metrics['quadratic_weighted_kappa']:.3f}")
        print("="*80)
        
        return model_path


if __name__ == "__main__":
    pipeline = FilteredDataTrainingPipeline()
    pipeline.run_training_pipeline()