"""Debug script to identify cross-validation issues."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from four_tier_classifier import HighConfidenceFourTierClassifier
from pathlib import Path

def load_and_combine_data():
    """Load and combine 2022 and 2023 data."""
    DATA_2022_PATH = "/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    DATA_2023_PATH = "/Users/JCR/Downloads/AdmissionsDataset/2023 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    
    print("Loading 2022 data...")
    pipeline_2022 = HighConfidenceFourTierClassifier(DATA_2022_PATH)
    df_2022 = pipeline_2022._prepare_data_for_training()
    
    print("Loading 2023 data...")
    pipeline_2023 = HighConfidenceFourTierClassifier(DATA_2023_PATH)
    df_2023 = pipeline_2023._prepare_data_for_training()
    
    # Combine datasets
    df_combined = pd.concat([df_2022, df_2023], ignore_index=True)
    
    # Create target mapping
    class_map = {label: idx for idx, label in enumerate(pipeline_2022.class_names)}
    df_combined["tier_int"] = df_combined["target"].map(class_map)
    
    print(f"Combined dataset shape: {df_combined.shape}")
    print(f"Class distribution:\n{df_combined['target'].value_counts()}")
    print(f"Class distribution (numeric):\n{df_combined['tier_int'].value_counts()}")
    
    return df_combined, pipeline_2022.features, class_map

def test_basic_cv():
    """Test basic cross-validation without custom wrappers."""
    df_combined, features, class_map = load_and_combine_data()
    
    X = df_combined[features].values  # Use numpy array
    y = df_combined["tier_int"].values
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {np.unique(y)}")
    print(f"Any NaN in X: {np.isnan(X).any()}")
    print(f"Any NaN in y: {np.isnan(y).any()}")
    
    # Test basic RandomForest
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Test simple cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nTesting basic cross-validation...")
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"CV scores: {scores}")
        print(f"Mean CV score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        
    # Test manual fit
    print("\nTesting manual fit...")
    try:
        model.fit(X, y)
        print("Manual fit successful")
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        print(f"Training accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Error in manual fit: {e}")

if __name__ == "__main__":
    test_basic_cv()
