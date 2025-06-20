"""
Train with Corrected Bucket Boundaries
======================================

Fix bucket assignment to match the actual score distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from ordinal_regression_model import OrdinalXGBoostClassifier


def assign_buckets(scores):
    """Assign scores to buckets with correct boundaries."""
    # Natural breaks in the data:
    # Reject: 0-9
    # Waitlist: 11-15  
    # Interview: 17-21
    # Accept: 23-25
    
    buckets = np.zeros(len(scores), dtype=int)
    
    # Reject bucket (0-9)
    buckets[scores <= 9] = 0
    
    # Waitlist bucket (11-15)
    buckets[(scores >= 11) & (scores <= 15)] = 1
    
    # Interview bucket (17-21)
    buckets[(scores >= 17) & (scores <= 21)] = 2
    
    # Accept bucket (23-25)
    buckets[scores >= 23] = 3
    
    return buckets


def load_all_data():
    """Load all filtered data with LLM scores."""
    print("\n1. Loading data...")
    
    # Load filtered data
    df_2022 = pd.read_excel("data_filtered/2022_filtered_applicants.xlsx")
    df_2023 = pd.read_excel("data_filtered/2023_filtered_applicants.xlsx") 
    df_2024 = pd.read_excel("data_filtered/2024_filtered_applicants.xlsx")
    
    print(f"   Loaded: 2022={len(df_2022)}, 2023={len(df_2023)}, 2024={len(df_2024)}")
    
    # Load LLM scores
    llm_train = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
    llm_train = llm_train.rename(columns={'AMCAS_ID_original': 'amcas_id'})
    
    llm_test_files = list(Path(".").glob("llm_scores_2024_*.csv"))
    if llm_test_files:
        llm_test = pd.read_csv(sorted(llm_test_files)[-1])
        llm_test = llm_test.rename(columns={'AMCAS_ID_original': 'amcas_id'})
    else:
        llm_test = None
    
    # Get LLM columns
    llm_cols = [col for col in llm_train.columns if col.startswith('llm_')]
    
    # Merge LLM scores
    df_2022 = df_2022.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    df_2023 = df_2023.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    
    if llm_test is not None:
        df_2024 = df_2024.merge(llm_test[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    
    # Combine training data
    df_train = pd.concat([df_2022, df_2023], ignore_index=True)
    
    return df_train, df_2024


def prepare_features(df, feature_cols=None):
    """Prepare features for modeling."""
    
    exclude_cols = ['application_review_score', 'amcas_id', 'appl_year', 'year', 'AMCAS_ID']
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Handle categorical features
    categorical_mappings = {
        'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1},
        'citizenship': {'US_Citizen': 0, 'Permanent_Resident': 1, 'International': 2, 'Other': 3},
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
    
    return X, feature_cols


def train_model(df_train, df_test):
    """Train ordinal regression model with corrected buckets."""
    print("\n2. Preparing features...")
    
    # Prepare features
    X_train, feature_cols = prepare_features(df_train)
    y_train = df_train['application_review_score'].values
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Assign buckets correctly
    y_train_buckets = assign_buckets(y_train)
    
    # Check bucket distribution
    print("\n   Training bucket distribution:")
    bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
    for i in range(4):
        count = (y_train_buckets == i).sum()
        pct = count / len(y_train_buckets) * 100
        print(f"   {bucket_names[i]}: {count} ({pct:.1f}%)")
    
    print("\n3. Training model with 5-fold CV...")
    
    # Use corrected boundaries for the model
    # These represent the thresholds between buckets
    bucket_boundaries = [0, 10, 16, 22, 26]
    
    # Create model
    model = OrdinalXGBoostClassifier(
        n_classes=4,
        bucket_boundaries=bucket_boundaries,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train_buckets)):
        X_fold_train = X_train_scaled[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train
        fold_model = OrdinalXGBoostClassifier(
            n_classes=4,
            bucket_boundaries=bucket_boundaries,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=fold
        )
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        
        # Predict
        pred_buckets, _ = fold_model.predict_with_confidence(X_fold_val)
        true_buckets = assign_buckets(y_fold_val)
        
        # Metrics
        exact = np.mean(pred_buckets == true_buckets)
        adjacent = np.mean(np.abs(pred_buckets - true_buckets) <= 1)
        qwk = cohen_kappa_score(true_buckets, pred_buckets, weights='quadratic')
        
        cv_results.append({'exact': exact, 'adjacent': adjacent, 'qwk': qwk})
        print(f"   Fold {fold+1}: Exact={exact:.3f}, Adjacent={adjacent:.3f}, QWK={qwk:.3f}")
    
    # Average results
    avg_exact = np.mean([r['exact'] for r in cv_results])
    avg_adjacent = np.mean([r['adjacent'] for r in cv_results]) 
    avg_qwk = np.mean([r['qwk'] for r in cv_results])
    
    print(f"\n   CV Average: Exact={avg_exact:.3f}, Adjacent={avg_adjacent:.3f}, QWK={avg_qwk:.3f}")
    
    # Train final model
    print("\n4. Training final model on all data...")
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Feature names for the model
    model.feature_names_ = feature_cols
    
    # Evaluate on test set
    print("\n5. Evaluating on 2024 test set...")
    X_test, _ = prepare_features(df_test, feature_cols)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    y_test = df_test['application_review_score'].values
    pred_test, conf_test = model.predict_with_confidence(X_test_scaled)
    true_test_buckets = assign_buckets(y_test)
    
    test_exact = np.mean(pred_test == true_test_buckets)
    test_adjacent = np.mean(np.abs(pred_test - true_test_buckets) <= 1)
    test_qwk = cohen_kappa_score(true_test_buckets, pred_test, weights='quadratic')
    
    print(f"   Test Exact={test_exact:.3f}, Adjacent={test_adjacent:.3f}, QWK={test_qwk:.3f}")
    
    # Confusion matrix
    print("\n   Confusion Matrix:")
    cm = confusion_matrix(true_test_buckets, pred_test)
    print("   True\\Pred  Reject  Waitlist  Interview  Accept")
    for i, name in enumerate(bucket_names):
        row = f"   {name:10}"
        for j in range(4):
            row += f"{cm[i,j]:8d}"
        print(row)
    
    # Prediction distribution
    pred_counts = pd.Series(pred_test).value_counts().sort_index()
    print("\n   Prediction distribution:")
    for i, name in enumerate(bucket_names):
        count = pred_counts.get(i, 0)
        pct = count / len(pred_test) * 100
        print(f"   - {name}: {count} ({pct:.1f}%)")
    
    # Feature importance
    print("\n6. Top 15 Feature Importance:")
    importance_dict = model.get_feature_importance()
    if importance_dict:
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance[:15]:
            print(f"   - {feature}: {importance:.3f}")
    
    # Save results
    results = pd.DataFrame({
        'amcas_id': df_test['amcas_id'],
        'true_score': y_test,
        'true_bucket': [bucket_names[i] for i in true_test_buckets],
        'predicted_bucket': [bucket_names[i] for i in pred_test],
        'confidence': conf_test,
        'service_rating': df_test['service_rating_numerical']
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f"predictions_corrected_{timestamp}.csv", index=False)
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'imputer': imputer,
        'scaler': scaler,
        'cv_metrics': {'exact': avg_exact, 'adjacent': avg_adjacent, 'qwk': avg_qwk},
        'test_metrics': {'exact': test_exact, 'adjacent': test_adjacent, 'qwk': test_qwk},
        'bucket_assignment_function': assign_buckets
    }
    
    model_path = f"models/ordinal_corrected_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    joblib.dump(model_data, "models/ordinal_corrected_latest.pkl")
    
    print(f"\n   Model saved to: {model_path}")
    
    return model, results


def main():
    print("="*80)
    print("TRAINING WITH CORRECTED BUCKET BOUNDARIES")
    print("="*80)
    
    # Load data
    df_train, df_test = load_all_data()
    
    # Train model
    model, results = train_model(df_train, df_test)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("Bucket boundaries fixed to match actual score distribution")
    print("="*80)


if __name__ == "__main__":
    main()