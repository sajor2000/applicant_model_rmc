"""
Train Model with Filtered Consistent Data - Fixed Version
========================================================

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
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """Load filtered training data and merge with LLM scores."""
        print("\n1. Loading and merging data...")
        
        # Load 2022 and 2023 filtered data
        df_2022 = pd.read_excel(self.filtered_data_path / "2022_filtered_applicants.xlsx")
        df_2023 = pd.read_excel(self.filtered_data_path / "2023_filtered_applicants.xlsx")
        
        # Combine
        df_train = pd.concat([df_2022, df_2023], ignore_index=True)
        print(f"   Loaded {len(df_2022)} from 2022, {len(df_2023)} from 2023")
        print(f"   Total training samples: {len(df_train)}")
        
        # Load LLM scores
        llm_files = list(Path(".").glob("llm_scores_2022_2023_*.csv"))
        if llm_files:
            latest_llm = sorted(llm_files)[-1]
            df_llm = pd.read_csv(latest_llm)
            print(f"   Loaded LLM scores from: {latest_llm}")
            
            # Rename AMCAS ID column
            if 'AMCAS_ID_original' in df_llm.columns:
                df_llm = df_llm.rename(columns={'AMCAS_ID_original': 'amcas_id'})
            elif 'AMCAS_ID' in df_llm.columns:
                df_llm = df_llm.rename(columns={'AMCAS_ID': 'amcas_id'})
            
            # Get LLM feature columns
            llm_cols = [col for col in df_llm.columns if col.startswith('llm_')]
            
            # Merge
            df_train = df_train.merge(
                df_llm[['amcas_id'] + llm_cols],
                on='amcas_id',
                how='left'
            )
            
            merged_count = df_train[llm_cols[0]].notna().sum()
            print(f"   Merged {len(llm_cols)} LLM features for {merged_count}/{len(df_train)} applicants")
        
        return df_train
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        print("\n2. Preparing features...")
        
        # Get target variable
        y = df['application_review_score'].values
        
        # Define feature columns (exclude target and ID)
        exclude_cols = ['application_review_score', 'amcas_id', 'appl_year', 
                       'AMCAS_ID', 'year']  # Also exclude any leftover ID/year columns
        
        # Get all numeric columns first
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Remove excluded columns
        numeric_features = [col for col in numeric_cols if col not in exclude_cols]
        categorical_features = [col for col in categorical_cols if col not in exclude_cols]
        
        print(f"   Found {len(numeric_features)} numeric features")
        print(f"   Found {len(categorical_features)} categorical features")
        
        # Create feature matrix
        X_numeric = df[numeric_features].copy()
        X_categorical = df[categorical_features].copy()
        
        # Handle categorical features
        cat_encoded = []
        cat_feature_names = []
        
        for col in categorical_features:
            if col == 'gender':
                X_categorical[col] = X_categorical[col].map({
                    'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1
                }).fillna(-1)
            elif col == 'citizenship':
                X_categorical[col] = X_categorical[col].map({
                    'US_Citizen': 0, 'Permanent_Resident': 1, 
                    'International': 2, 'Other': 3
                }).fillna(3)
            elif col == 'service_rating_categorical':
                rating_map = {
                    'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                    'Good': 2, 'Average': 1, 'Below Average': 0,
                    'Poor': -1
                }
                X_categorical[col] = X_categorical[col].map(rating_map).fillna(-2)
            else:
                # For other categorical, use simple encoding
                X_categorical[col] = pd.Categorical(X_categorical[col]).codes
            
            cat_encoded.append(X_categorical[col].values.reshape(-1, 1))
            cat_feature_names.append(col)
        
        # Combine numeric and encoded categorical
        if cat_encoded:
            X_cat_array = np.hstack(cat_encoded)
            X_combined = np.hstack([X_numeric.values, X_cat_array])
            feature_names = numeric_features + cat_feature_names
        else:
            X_combined = X_numeric.values
            feature_names = numeric_features
        
        print(f"   Total features after encoding: {X_combined.shape[1]}")
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_combined)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Debug shape mismatch
        print(f"   X_scaled shape: {X_scaled.shape}")
        print(f"   Feature names length: {len(feature_names)}")
        
        # Ensure shapes match
        if X_scaled.shape[1] != len(feature_names):
            print(f"   WARNING: Shape mismatch! Adjusting feature names...")
            # Use generic feature names if mismatch
            feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        
        # Create DataFrame with all features
        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        return X_df, y, feature_names, imputer, scaler
    
    def train_ordinal_model(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """Train ordinal regression model with cross-validation."""
        print("\n3. Training ordinal regression model...")
        
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
            pred_buckets, _ = model.predict_with_confidence(X_val)
            true_buckets = np.digitize(y_val, bucket_boundaries[1:-1]) - 1
            
            # Calculate metrics
            exact_match = np.mean(pred_buckets == true_buckets)
            
            # Adjacent accuracy (within 1 bucket)
            diff = np.abs(pred_buckets - true_buckets)
            adjacent = np.mean(diff <= 1)
            
            # Quadratic weighted kappa
            from sklearn.metrics import cohen_kappa_score
            qwk = cohen_kappa_score(true_buckets, pred_buckets, weights='quadratic')
            
            cv_scores.append(exact_match)
            cv_adjacent.append(adjacent)
            cv_qwk.append(qwk)
            
            print(f"   Fold {fold+1}: Exact={exact_match:.3f}, "
                  f"Adjacent={adjacent:.3f}, "
                  f"QWK={qwk:.3f}")
        
        print(f"\n   Average CV Performance:")
        print(f"   - Exact Match: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
        print(f"   - Adjacent: {np.mean(cv_adjacent):.3f} (±{np.std(cv_adjacent):.3f})")
        print(f"   - QWK: {np.mean(cv_qwk):.3f} (±{np.std(cv_qwk):.3f})")
        
        # Train final model on all data
        print("\n4. Training final model on all data...")
        model.fit(X, y)
        
        return model, {
            'cv_exact_match': np.mean(cv_scores),
            'cv_adjacent': np.mean(cv_adjacent),
            'cv_qwk': np.mean(cv_qwk)
        }
    
    def evaluate_on_2024(self, model, feature_names: list, 
                        imputer, scaler) -> tuple:
        """Evaluate model on 2024 holdout data."""
        print("\n5. Evaluating on 2024 holdout data...")
        
        # Load 2024 data
        df_2024 = pd.read_excel(self.filtered_data_path / "2024_filtered_applicants.xlsx")
        print(f"   Loaded {len(df_2024)} test samples from 2024")
        
        # Load LLM scores for 2024 if available
        llm_files_2024 = list(Path(".").glob("llm_scores_2024_*.csv"))
        if llm_files_2024:
            latest_llm = sorted(llm_files_2024)[-1]
            df_llm = pd.read_csv(latest_llm)
            
            if 'AMCAS_ID_original' in df_llm.columns:
                df_llm = df_llm.rename(columns={'AMCAS_ID_original': 'amcas_id'})
            elif 'AMCAS_ID' in df_llm.columns:
                df_llm = df_llm.rename(columns={'AMCAS_ID': 'amcas_id'})
            
            llm_cols = [col for col in df_llm.columns if col.startswith('llm_')]
            df_2024 = df_2024.merge(
                df_llm[['amcas_id'] + llm_cols],
                on='amcas_id',
                how='left'
            )
            print(f"   Merged LLM scores for 2024 data")
        
        # Prepare features exactly as in training
        exclude_cols = ['application_review_score', 'amcas_id', 'appl_year', 
                       'AMCAS_ID', 'year']
        
        # Separate numeric and categorical based on feature names
        numeric_features = [f for f in feature_names if f in df_2024.columns and 
                           df_2024[f].dtype in [np.number, 'int64', 'float64']]
        categorical_features = [f for f in feature_names if f in df_2024.columns and 
                               f not in numeric_features]
        
        # Handle missing features
        X_test_list = []
        
        for feature in feature_names:
            if feature in df_2024.columns:
                if feature in categorical_features:
                    # Apply same encoding as training
                    if feature == 'gender':
                        values = df_2024[feature].map({
                            'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1
                        }).fillna(-1)
                    elif feature == 'citizenship':
                        values = df_2024[feature].map({
                            'US_Citizen': 0, 'Permanent_Resident': 1, 
                            'International': 2, 'Other': 3
                        }).fillna(3)
                    elif feature == 'service_rating_categorical':
                        rating_map = {
                            'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                            'Good': 2, 'Average': 1, 'Below Average': 0,
                            'Poor': -1
                        }
                        values = df_2024[feature].map(rating_map).fillna(-2)
                    else:
                        values = pd.Categorical(df_2024[feature]).codes
                else:
                    values = df_2024[feature]
                
                X_test_list.append(values.values.reshape(-1, 1))
            else:
                # Feature not in test data, use zeros
                X_test_list.append(np.zeros((len(df_2024), 1)))
                print(f"   Warning: Feature '{feature}' not found in 2024 data, using zeros")
        
        X_test = np.hstack(X_test_list)
        
        # Apply same preprocessing
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
        
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
        true_buckets = np.digitize(y_true, [0, 10, 16, 22, 26][1:-1]) - 1
        
        # Calculate metrics
        exact_match = np.mean(predictions == true_buckets)
        diff = np.abs(predictions - true_buckets)
        adjacent = np.mean(diff <= 1)
        
        from sklearn.metrics import cohen_kappa_score
        qwk = cohen_kappa_score(true_buckets, predictions, weights='quadratic')
        
        metrics = {
            'exact_match_accuracy': exact_match,
            'adjacent_accuracy': adjacent,
            'quadratic_weighted_kappa': qwk
        }
        
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
    
    def save_model(self, model, feature_names: list, imputer, scaler, 
                  cv_metrics: dict, test_metrics: dict):
        """Save trained model and preprocessing objects."""
        print("\n6. Saving model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_save_path / f"ordinal_model_filtered_{timestamp}.pkl"
        
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'imputer': imputer,
            'scaler': scaler,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'training_date': timestamp,
            'n_features': len(feature_names),
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
        
        # Load and merge data
        df_train = self.load_and_merge_data()
        
        # Prepare features
        X, y, feature_names, imputer, scaler = self.prepare_features(df_train)
        
        # Train model
        model, cv_metrics = self.train_ordinal_model(X, y)
        
        # Evaluate on 2024
        results_2024, test_metrics = self.evaluate_on_2024(
            model, feature_names, imputer, scaler
        )
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"predictions_2024_filtered_{timestamp}.csv"
        results_2024.to_csv(results_path, index=False)
        print(f"\n   Predictions saved to: {results_path}")
        
        # Save model
        model_path = self.save_model(
            model, feature_names, imputer, scaler, cv_metrics, test_metrics
        )
        
        # Feature importance
        print("\n7. Top Feature Importance:")
        importance_df = model.get_feature_importance(feature_names)
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