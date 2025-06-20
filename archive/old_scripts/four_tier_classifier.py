import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class HighConfidenceFourTierClassifier:
    def __init__(self, applicants_file_path):
        """Initializes the classifier by loading the primary applicant data."""
        self.applicants_file_path = applicants_file_path
        self.raw_df = None
        self.processed_df = None
        self.features = None
        self.model = None
        self.scaler = None
        self.class_names = None
        self.model_save_path = 'models/four_tier_high_confidence_model.pkl'

        try:
            self.raw_df = pd.read_excel(self.applicants_file_path)
            print(f"Successfully loaded data from {self.applicants_file_path}")
        except FileNotFoundError:
            print(f"Error: Applicants file not found at {self.applicants_file_path}")
            raise
        
        self._ensure_models_directory()

    def _ensure_models_directory(self):
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Created 'models/' directory.")

    def _handle_missing_data_and_ids(self, df):
        """Handles ID standardization and missing data as per guide specifications."""
        # Standardize AMCAS ID column and type
        if 'Amcas_ID' in df.columns and 'AMCAS ID' not in df.columns:
            df.rename(columns={'Amcas_ID': 'AMCAS ID'}, inplace=True)
        elif 'AMCAS ID' not in df.columns and 'Amcas_ID' not in df.columns:
            print("Warning: Neither 'AMCAS ID' nor 'Amcas_ID' found. Cannot standardize ID.")
            # Consider raising an error if ID is critical and missing
        
        if 'AMCAS ID' in df.columns:
            df['AMCAS ID'] = df['AMCAS ID'].astype(str)

        # Fill numeric with 0, GPA 'NULL' to 0
        numeric_cols_to_fill_zero = [
            'Exp_Hour_Total', 'Exp_Hour_Research', 'Exp_Hour_Volunteer_Med', 
            'Exp_Hour_Shadowing', 'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Age', 'Num_Dependents', 'Service Rating (Numerical)'
        ]
        for col in numeric_cols_to_fill_zero:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                print(f"Warning: Numeric column '{col}' for zero-filling not found. Creating and filling with 0.")
                df[col] = 0

        gpa_trend_cols = ['Total_GPA_Trend', 'BCPM_GPA_Trend']
        for col in gpa_trend_cols:
            if col in df.columns:
                df[col] = df[col].replace('NULL', 0).fillna(0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Ensure numeric after replace
            else:
                print(f"Warning: GPA trend column '{col}' not found. Creating and filling with 0.")
                df[col] = 0
        return df

    def _engineer_features(self, df):
        """Creates derived features as specified in the implementation guide."""
        epsilon = 1e-6 # To avoid division by zero

        # Ensure required columns exist, defaulting to 0 if not (after _handle_missing_data_and_ids)
        required_for_eng = {
            'Exp_Hour_Research': 0, 'Exp_Hour_Total': 0, 'Exp_Hour_Volunteer_Med': 0,
            'Exp_Hour_Shadowing': 0, 'Service Rating (Numerical)':0, 
            'Comm_Service_Total_Hours':0, 'Disadvantanged_Ind':'No', # Default for binary conversion later
            'Total_GPA_Trend':0
        }
        for col, default_val in required_for_eng.items():
            if col not in df.columns:
                print(f"Warning: Column '{col}' for feature engineering not found. Defaulting to {default_val}.")
                df[col] = default_val
                if isinstance(default_val, str): # Ensure correct type for subsequent ops
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)

        df['research_intensity'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
        df['clinical_intensity'] = (df['Exp_Hour_Volunteer_Med'] + df['Exp_Hour_Shadowing']) / (df['Exp_Hour_Total'] + epsilon)
        # Definition of experience_balance: Ratio of research to sum of key clinical hours
        df['experience_balance'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Volunteer_Med'] + df['Exp_Hour_Shadowing'] + epsilon)
        # Ensure Comm_Service_Total_Hours is non-negative before log
        df['Comm_Service_Total_Hours_Positive'] = df['Comm_Service_Total_Hours'].apply(lambda x: max(x, 0))
        df['service_commitment'] = df['Service Rating (Numerical)'] * np.log(df['Comm_Service_Total_Hours_Positive'] + 1) 
        
        # Adversity_overcome: Disadvantaged_Ind (binary) * Total_GPA_Trend (proxy for achievements)
        # Disadvantaged_Ind will be converted to binary later. Assuming 'Yes' maps to 1.
        # For now, create a placeholder that will be correct after Disadvantaged_Ind is binarized.
        # This assumes Total_GPA_Trend is a suitable numeric proxy for achievements.
        # If Disadvantaged_Ind is 'Yes', this will be 1 * GPA_Trend, if 'No', 0 * GPA_Trend.
        if 'Disadvantanged_Ind' in df.columns and isinstance(df['Disadvantanged_Ind'].iloc[0], str):
             df['adversity_overcome_temp_ind'] = df['Disadvantanged_Ind'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
             df['adversity_overcome'] = df['adversity_overcome_temp_ind'] * df['Total_GPA_Trend']
             df.drop(columns=['adversity_overcome_temp_ind'], inplace=True)
        else: # If Disadvantaged_Ind is already numeric or missing
            df['Disadvantanged_Ind_numeric'] = pd.to_numeric(df['Disadvantanged_Ind'], errors='coerce').fillna(0)
            df['adversity_overcome'] = df['Disadvantanged_Ind_numeric'] * df['Total_GPA_Trend']
            if 'Disadvantanged_Ind_numeric' in df.columns: df.drop(columns=['Disadvantanged_Ind_numeric'], inplace=True)

        print("Derived features engineered.")
        return df

    def _define_target_and_features(self, df):
        """Defines the target variable and final feature list."""
        # Define target tiers
        if 'Application Review Score' not in df.columns:
            print("Error: 'Application Review Score' not found. Cannot create target variable.")
            df['target'] = 'Error'
            self.class_names = []
            return df, [] # Return empty features list
        
        def assign_tier(score):
            if pd.isna(score): return None
            score = pd.to_numeric(score, errors='coerce') # Ensure score is numeric
            if pd.isna(score): return None
            if score <= 14: return '1. Very Unlikely'
            if 15 <= score <= 18: return '2. Potential Review'
            if 19 <= score <= 22: return '3. Probable Interview'
            return '4. Very Likely Interview'

        df['target'] = df['Application Review Score'].apply(assign_tier)
        df.dropna(subset=['target'], inplace=True)
        self.class_names = sorted(list(df['target'].unique()))
        print("Target variable created.")

        # Convert core categorical features to binary (Yes->1, else 0)
        binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
        binary_conversion_map = {'Yes': 1, 'yes': 1} # Handle potential case variations
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).map(lambda x: binary_conversion_map.get(x.lower(), 0)).astype(int)
            else:
                print(f"Warning: Binary column '{col}' not found. Creating and filling with 0.")
                df[col] = 0
        print("Binary indicator features converted.")

        # Define final feature list (excluding ID, target, and raw score)
        # Includes original critical columns + new engineered features + binary indicators
        base_features = [
            'Age', 'Num_Dependents',
            'Exp_Hour_Total', 'Exp_Hour_Research', 'Exp_Hour_Volunteer_Med', 
            'Exp_Hour_Shadowing', 'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Service Rating (Numerical)',
            'Total_GPA_Trend', 'BCPM_GPA_Trend',
            'Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value' 
        ]
        engineered_features = [
            'research_intensity', 'clinical_intensity', 'experience_balance', 
            'service_commitment', 'adversity_overcome'
        ]
        final_features = [col for col in base_features + engineered_features if col in df.columns]
        
        # Ensure all features are numeric and handle any NaNs that might have slipped through or been created
        for col in final_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        self.features = final_features
        print(f"Final features for model: {self.features}")
        return df

    def _prepare_data_for_training(self):
        """Full pipeline for data prep before training."""
        df = self.raw_df.copy()
        df = self._handle_missing_data_and_ids(df)
        df = self._engineer_features(df)
        df = self._define_target_and_features(df)
        self.processed_df = df
        return df

    def train_ensemble_with_calibration(self):
        """Trains the model ensemble and applies calibration (Placeholder)."""
        if self.processed_df is None:
            self._prepare_data_for_training()
        
        if 'target' not in self.processed_df.columns or self.processed_df['target'].nunique() < 2:
            print("Error: Target variable not suitable for training. Aborting training.")
            return

        X = self.processed_df[self.features]
        y = self.processed_df['target']

                # Use a 70-30 train-test split as requested for more robust evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # Scale experience and other high-variance numerical features
        # Identify columns to scale (e.g., hours, trends, possibly engineered features)
        cols_to_scale = [col for col in self.features if 'Hour' in col or 'Hours' in col or 'Trend' in col or 
                         col in ['Age', 'Num_Dependents', 'research_intensity', 'clinical_intensity', 
                                  'experience_balance', 'service_commitment', 'adversity_overcome']]
        # Filter out binary columns that might have 'Ind' but are not hours/trends
        cols_to_scale = [col for col in cols_to_scale if col not in ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']]
        
        self.scaler = StandardScaler()
        if cols_to_scale:
            print(f"Scaling features: {cols_to_scale}")
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        else:
            print("No features identified for scaling or scaler already fitted.")
            X_train_scaled = X_train
            X_test_scaled = X_test
            self.scaler = None # No scaler if no columns scaled

        print("Training RandomForestClassifier (as part of ensemble placeholder)...")
        # Placeholder for more complex ensemble and calibration
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)
        print("Model training complete.")

        y_pred = self.model.predict(X_test_scaled)
        print("--- Classification Report (on test set) ---")
        print(classification_report(y_test, y_pred, zero_division=0, target_names=self.class_names))
        print("\n--- Confusion Matrix (on test set) ---")
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)
        print(pd.DataFrame(cm, index=self.class_names, columns=self.class_names))
        # Further calibration steps would go here
        print("Ensemble training and calibration placeholder complete.")

    def create_production_model(self):
        """Saves the trained model, features, scaler, and class names for production."""
        if not self.model or not self.features or not self.class_names:
            print("Error: Model not trained or essential components missing. Cannot save production model.")
            print("Run train_ensemble_with_calibration() first.")
            return

        model_data_to_save = {
            'model': self.model,
            'feature_names': self.features,
            'scaler': self.scaler, 
            'class_names': self.class_names
        }
        joblib.dump(model_data_to_save, self.model_save_path)
        print(f"Production model data saved to '{self.model_save_path}'.")

# Example Usage (for testing the class structure)
if __name__ == '__main__':
    # Path to your main applicants Excel file
    # Ensure this file is in the correct location relative to where you run the script
    # or provide an absolute path.
    # For example: '/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx'
    APPLICANTS_FILE = "/Users/JCR/Downloads/AdmissionsDataset/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    
    print(f"Attempting to load data from: {APPLICANTS_FILE}")
    if not os.path.exists(APPLICANTS_FILE):
        print(f"CRITICAL ERROR: The file {APPLICANTS_FILE} does not exist. Please check the path.")
    else:
        try:
            classifier = HighConfidenceFourTierClassifier(APPLICANTS_FILE)
            print("\n--- Preparing data for training ---")
            classifier._prepare_data_for_training() # Call the prep method
            if classifier.processed_df is not None and not classifier.processed_df.empty:
                print("\n--- Processed DataFrame head: ---")
                print(classifier.processed_df.head())
                print("\n--- Training ensemble with calibration ---")
                classifier.train_ensemble_with_calibration()
                print("\n--- Creating production model ---")
                classifier.create_production_model()
                print("\nProcess completed. Check for 'models/four_tier_high_confidence_model.pkl'.")
            else:
                print("Data processing failed, cannot proceed with training.")
        except Exception as e:
            print(f"An error occurred during the process: {e}")
            import traceback
            print(traceback.format_exc())
