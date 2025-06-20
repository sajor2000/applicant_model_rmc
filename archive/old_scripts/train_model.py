import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
import os

# Define the path to the dataset
DATA_PATH = "/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/"
from sklearn.preprocessing import StandardScaler # Added for future use

def preprocess_data():
    """Loads and preprocesses the applicant data."""
    print("Loading data...")
    # Load data primarily from 1. Applicants.xlsx
    df = pd.read_excel(f"{DATA_PATH}1. Applicants.xlsx")
    print("Data loaded successfully from 1. Applicants.xlsx.")

    # --- Debug: Print column names --- 
    print("\n--- Columns in 1. Applicants.xlsx ---")
    print(df.columns)

    # --- Inspect key categorical columns for IntegratedPredictor compatibility ---
    categorical_cols_to_inspect = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
    print("\n--- Unique values in key categorical columns ---")
    for col_name in categorical_cols_to_inspect:
        if col_name in df.columns:
            print(f"Unique values in '{col_name}': {df[col_name].unique()}")
        else:
            print(f"Warning: Column '{col_name}' not found in DataFrame.")

    # --- Feature Engineering & Selection ---
    # Using 'Total_GPA_Trend' directly from '1. Applicants.xlsx'
    # 'MCAT_Total_Score' is still pending identification in the dataset.
    features = [
        'Age',
        'Exp_Hour_Total',
        'Exp_Hour_Research',
        'Exp_Hour_Volunteer_Med',
        'Comm_Service_Total_Hours',
        'HealthCare_Total_Hours',
        'MCAT_Total_Score', 
        'Total_GPA_Trend', 
        # Adding categorical features that IntegratedPredictor expects (will be converted to binary later)
        'Disadvantanged_Ind',
        'First_Generation_Ind',
        'SES_Value'
    ]
    
    # Ensure 'AMCAS ID' is present for potential future use, prefer 'Amcas_ID' if 'AMCAS ID' is not there.
    if 'AMCAS ID' not in df.columns and 'Amcas_ID' in df.columns:
        df.rename(columns={'Amcas_ID': 'AMCAS ID'}, inplace=True)
        print("Renamed 'Amcas_ID' to 'AMCAS ID' in df.")

    # Impute missing values for selected features
    for col in features: # Iterate over the defined features list
        if col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else: # For categorical features
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown') # Impute with mode or 'Unknown'
        else:
            print(f"Warning: Feature column '{col}' not found. Filling with 0 or 'Unknown'.")
            if 'Hour' in col or 'GPA' in col or 'Age' in col or 'MCAT' in col: # Numeric defaults
                 df[col] = 0
            else: # Categorical defaults (e.g., for _Ind columns)
                 df[col] = 'No' 

    # Convert specified categorical features to binary (0/1)
    # These are the features IntegratedAdmissionsPredictor expects to be binary.
    binary_conversion_map = {'Yes': 1, 'No': 0}
    # For SES_Value, assuming it might have numeric-like strings or other values.
    # A more robust mapping or binning might be needed if SES_Value is not simply 'Yes'/'No' or equivalent.
    # For now, treat 'Yes' as 1, others as 0 for simplicity matching IntegratedAdmissionsPredictor's current logic.
    # If SES_Value is actually numeric (e.g. 1-5), this conversion needs to be rethought.
    # Based on previous inspection, SES_Value seems to be 'Yes'/'No' or similar like other _Ind columns.
    
    categorical_to_binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
    for col_name in categorical_to_binary_cols:
        if col_name in df.columns:
            # Ensure all values are strings before mapping, to handle mixed types like True/False/NaN
            df[col_name] = df[col_name].astype(str).map(binary_conversion_map).fillna(0).astype(int)
            print(f"Converted '{col_name}' to binary. Unique values: {df[col_name].unique()}")
        else:
            # If column was created as a default (e.g. 'No'), it's already effectively 0 after mapping.
            # If it was numeric (e.g. 0), it will also be 0.
            # This handles cases where a feature might be missing and defaulted.
            print(f"Warning: Column '{col_name}' for binary conversion not found, was likely defaulted.")
            # Ensure it exists and is 0 if it was defaulted during imputation.
            if col_name not in df.columns:
                 df[col_name] = 0 
            else: # If it exists but wasn't 'Yes', ensure it's 0
                 df[col_name] = df[col_name].astype(str).map(binary_conversion_map).fillna(0).astype(int)


    print("Feature engineering, imputation, and binary conversion complete.")

    # --- Define Target Variable ---
    # Ensure 'Application Review Score' exists
    if 'Application Review Score' not in df.columns:
        print("Error: 'Application Review Score' not found. Cannot create target variable.")
        # Potentially return None or raise error to stop processing
        return None, None 

    def assign_tier(score):
        if pd.isna(score): return None # Handle NaN scores before comparison
        if score <= 14: return '1. Very Unlikely'
        if 15 <= score <= 18: return '2. Potential Review'
        if 19 <= score <= 22: return '3. Probable Interview'
        return '4. Very Likely Interview'

    df['target'] = df['Application Review Score'].apply(assign_tier)
    df.dropna(subset=['target'], inplace=True) # Remove rows where target could not be assigned
    print("Target variable created.")
    
    return df, features

def train_and_evaluate(df, features):
    """Trains a model and evaluates its performance."""
    X = df[features]
    y = df['target']
    class_names = sorted(list(y.unique())) # Get class names for saving

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # --- Scale Experience Hour Features ---
    experience_cols = [col for col in features if 'Hour' in col or 'Hours' in col]
    scaler = StandardScaler()

    if experience_cols:
        print(f"Scaling experience columns: {experience_cols}")
        # Fit scaler ONLY on training data
        X_train[experience_cols] = scaler.fit_transform(X_train[experience_cols])
        # Transform test data using the SAME fitted scaler
        X_test[experience_cols] = scaler.transform(X_test[experience_cols])
        print("Experience columns scaled.")
    else:
        print("No experience columns found to scale.")
        scaler = None # No scaler to save if no columns were scaled

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=class_names))

    print("\n--- Confusion Matrix ---")
    # Ensure labels for confusion matrix match the sorted class_names for consistent display
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print(pd.DataFrame(cm, index=class_names, columns=class_names))


    # --- Save Model, Features, Scaler, and Class Names ---
    model_filename = 'models/admissions_model_four_tier.pkl' # Aligned with IntegratedPredictor expectation (adjust if 3/4 tier changes)
    model_data_to_save = {
        'model': model,
        'feature_names': features,
        'scaler': scaler, # Save the fitted scaler
        'class_names': class_names # Save the class names
    }

    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models/' directory.")

    joblib.dump(model_data_to_save, model_filename)
    print(f"\nModel data (model, features, scaler, class_names) saved to '{model_filename}'.")

if __name__ == "__main__":
    # Redirect output to a log file and also handle errors
    with open('training_results.txt', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            print("Starting model training process...")
            processed_df, model_features = preprocess_data()
            train_and_evaluate(processed_df, model_features)
            print("Process finished successfully.")
        except Exception as e:
            print("\n--- AN ERROR OCCURRED ---")
            print(str(e))
            import traceback
            traceback.print_exc()
    
    # Restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Training complete. Results saved to training_results.txt")
