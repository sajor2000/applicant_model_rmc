"""
Data Transformation Pipeline for Medical Admissions
==================================================

Clean, staged approach letting tree-based models find patterns naturally.
No explicit interaction features - trees will discover these.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataTransformationPipeline:
    """
    Transforms raw admissions data through clear stages:
    1. Load and merge data
    2. Handle missing values
    3. Create derived features (ratios only)
    4. Encode categorical variables
    5. Scale numeric features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_mappings = {}
        self.feature_names = []
        
    def stage1_load_and_merge(self, base_path: str, year: int) -> pd.DataFrame:
        """
        Stage 1: Load all Excel files and merge on AMCAS_ID
        """
        logger.info(f"Stage 1: Loading {year} data")
        year_path = Path(base_path) / f"{year} Applicants Reviewed by Trusted Reviewers"
        
        # Primary data source
        df = pd.read_excel(year_path / "1. Applicants.xlsx")
        df['AMCAS_ID'] = df['Amcas_ID'].astype(str)
        
        # Add language features
        try:
            lang_df = pd.read_excel(year_path / "2. Language.xlsx")
            lang_features = lang_df.groupby('AMCAS_ID').agg({
                'Language': 'count',
                'Prof_Level': lambda x: (x == 'Native').sum()
            }).rename(columns={
                'Language': 'num_languages',
                'Prof_Level': 'num_native_languages'
            })
            df = df.merge(lang_features, left_on='AMCAS_ID', right_index=True, how='left')
        except:
            logger.warning("Could not load language data")
            
        # Add parent education (for first-gen status)
        try:
            parent_df = pd.read_excel(year_path / "3. Parents.xlsx")
            # Check if any parent has bachelor's or higher
            parent_education = parent_df.groupby('AMCAS_ID').agg({
                'Edu_Level': lambda x: any(edu in ['Bachelor', 'Master', 'Doctorate', 'Professional'] 
                                         for edu in x)
            }).rename(columns={'Edu_Level': 'parent_has_degree'})
            df = df.merge(parent_education, left_on='AMCAS_ID', right_index=True, how='left')
        except:
            logger.warning("Could not load parent data")
            
        # Add sibling count
        try:
            sibling_df = pd.read_excel(year_path / "4. Siblings.xlsx")
            sibling_count = sibling_df.groupby('AMCAS_ID').size().rename('num_siblings')
            df = df.merge(sibling_count, left_on='AMCAS_ID', right_index=True, how='left')
        except:
            logger.warning("Could not load sibling data")
            
        # Add calculated GPAs from academic records
        try:
            academic_df = pd.read_excel(year_path / "5. Academic Records.xlsx")
            gpa_calc = academic_df.groupby('AMCAS_ID').apply(
                lambda x: pd.Series({
                    'calculated_total_gpa': (x['Credit_Hrs'] * x['GPA_Points']).sum() / x['Credit_Hrs'].sum() 
                                          if x['Credit_Hrs'].sum() > 0 else 0,
                    'total_credit_hours': x['Credit_Hrs'].sum()
                })
            )
            df = df.merge(gpa_calc, left_on='AMCAS_ID', right_index=True, how='left')
        except:
            logger.warning("Could not load academic data")
            
        logger.info(f"Stage 1 complete: {len(df)} applicants loaded")
        return df
    
    def stage2_handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Handle missing values systematically
        """
        logger.info("Stage 2: Handling missing values")
        df = df.copy()
        
        # Numeric features - fill with 0 (represents no experience)
        numeric_fill_zero = [
            'Exp_Hour_Total', 'Exp_Hour_Research', 'Exp_Hour_Volunteer_Med',
            'Exp_Hour_Volunteer_Non_Med', 'Exp_Hour_Employ_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'num_languages', 'num_native_languages', 'num_siblings'
        ]
        
        for col in numeric_fill_zero:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # GPA trends - handle 'NULL' string
        for col in ['Total_GPA_Trend', 'BCPM_GPA_Trend']:
            if col in df.columns:
                df[col] = df[col].replace('NULL', 0).fillna(0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Financial percentages - fill with median
        financial_cols = [
            'Academic_Scholarship_Percentage', 'Finacial_Need_Based_Percentage',
            'Student_Loan_Percentage', 'Family_Contribution_Percentage'
        ]
        for col in financial_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Categorical - fill with 'Unknown'
        categorical_cols = ['Gender', 'Citizenship', 'Major_Long_Desc', 'Under_School']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        logger.info("Stage 2 complete: Missing values handled")
        return df
    
    def stage3_create_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Create simple ratio features (no interactions)
        """
        logger.info("Stage 3: Creating ratio features")
        df = df.copy()
        epsilon = 1e-6
        
        # Experience proportions (what % of time in each activity)
        if 'Exp_Hour_Total' in df.columns and df['Exp_Hour_Total'].sum() > 0:
            df['research_proportion'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
            df['clinical_proportion'] = (df['Exp_Hour_Volunteer_Med'] + df.get('Exp_Hour_Shadowing', 0)) / (df['Exp_Hour_Total'] + epsilon)
            df['volunteer_proportion'] = df.get('Exp_Hour_Volunteer_Non_Med', 0) / (df['Exp_Hour_Total'] + epsilon)
            df['employment_proportion'] = df.get('Exp_Hour_Employ_Med', 0) / (df['Exp_Hour_Total'] + epsilon)
        
        # Simple diversity indicator count
        diversity_indicators = [
            'First_Generation_Ind', 'Disadvantanged_Ind', 'RU_Ind',
            'Pell_Grant', 'Fee_Assistance_Program', 'Family_Assistance_Program'
        ]
        
        # Convert to binary and sum
        df['diversity_indicator_count'] = 0
        for indicator in diversity_indicators:
            if indicator in df.columns:
                df['diversity_indicator_count'] += (df[indicator] == 'Yes').astype(int)
        
        # Age-normalized experience (experience per year since 18)
        if 'Age' in df.columns and 'Exp_Hour_Total' in df.columns:
            df['experience_per_year'] = df['Exp_Hour_Total'] / (df['Age'] - 18 + epsilon)
        
        logger.info("Stage 3 complete: Ratio features created")
        return df
    
    def stage4_encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Stage 4: Convert categorical variables to numeric
        """
        logger.info("Stage 4: Encoding categorical variables")
        df = df.copy()
        
        # Binary Yes/No conversions
        binary_columns = [
            'First_Generation_Ind', 'Disadvantanged_Ind', 'RU_Ind',
            'Pell_Grant', 'Fee_Assistance_Program', 'Childhood_Med_Underserved_Self_Reported',
            'Family_Assistance_Program', 'Paid_Employment_BF_18',
            'Prev_Applied_Rush', 'Inst_Action_Ind', 'Prev_Matric_Ind',
            'Investigation_Ind', 'Felony_Ind', 'Misdemeanor_Ind',
            'Military_Discharge_Ind', 'Comm_Service_Ind', 'HealthCare_Ind'
        ]
        
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        
        # Ordinal encoding for service rating
        if 'Service Rating (Categorical)' in df.columns:
            service_map = {
                'Lacking/Does Not Meet': 1,
                'Adequate': 2,
                'Significant': 3,
                'Exceptional': 4
            }
            df['service_rating_encoded'] = df['Service Rating (Categorical)'].map(service_map).fillna(2)
        
        # One-hot encoding for key categoricals (limit to top categories to avoid explosion)
        one_hot_columns = {
            'Gender': 5,  # Max 5 categories
            'Citizenship': 10,  # Top 10 citizenships
            'SES_Value': 5,
            'Major_Long_Desc': 20,  # Top 20 majors
        }
        
        for col, max_categories in one_hot_columns.items():
            if col in df.columns:
                if fit:
                    # Get top N categories
                    top_categories = df[col].value_counts().head(max_categories).index.tolist()
                    self.categorical_mappings[col] = top_categories
                
                # Create binary columns for top categories
                for category in self.categorical_mappings.get(col, []):
                    df[f'{col}_{category}'] = (df[col] == category).astype(int)
                
                # Drop original column
                df = df.drop(columns=[col])
        
        # Convert family income to ordinal scale
        if 'Family_Income_Level' in df.columns:
            income_map = {
                'Less than $25,000': 1, '$25,000 - $29,999': 2,
                '$30,000 - $39,999': 3, '$40,000 - $49,999': 4,
                '$50,000 - $59,999': 5, '$60,000 - $74,999': 6,
                '$75,000 - $99,999': 7, '$100,000 - $124,999': 8,
                '$125,000 - $149,999': 9, '$150,000 - $174,999': 10,
                '$175,000 - $199,999': 11, '$200,000 - $249,999': 12,
                '$250,000 - $299,999': 13, '$300,000 - $349,999': 14,
                '$350,000 - $399,999': 15, '$400,000 or more': 16,
                "Don't know": 8, "Decline to respond": 8  # Median
            }
            df['family_income_ordinal'] = df['Family_Income_Level'].map(income_map).fillna(8)
            df = df.drop(columns=['Family_Income_Level'])
        
        logger.info("Stage 4 complete: Categoricals encoded")
        return df
    
    def stage5_select_and_scale(self, df: pd.DataFrame, 
                               llm_scores: pd.DataFrame = None,
                               fit_scaler: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Stage 5: Merge LLM scores, select features, and scale
        """
        logger.info("Stage 5: Final feature selection and scaling")
        
        # Merge LLM scores if provided
        if llm_scores is not None:
            df = df.merge(llm_scores, on='AMCAS_ID', how='left')
            
            # Fill missing LLM scores with neutral values
            llm_columns = [col for col in llm_scores.columns if col.startswith('llm_')]
            for col in llm_columns:
                if col.endswith('_score'):
                    df[col] = df[col].fillna(5)  # Neutral score
                elif col == 'llm_red_flag_count':
                    df[col] = df[col].fillna(0)
                elif col == 'llm_green_flag_count':
                    df[col] = df[col].fillna(0)
        
        # Remove non-feature columns
        exclude_columns = [
            'AMCAS_ID', 'Amcas_ID', 'Application Review Score',
            'Name', 'Email', 'Phone', 'Address',
            'Service Rating (Categorical)',  # Already encoded
            'Service Rating (Numerical)'  # Use encoded version
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                logger.warning(f"Dropping non-numeric column: {col}")
        
        # Select features
        X = df[numeric_features].copy()
        
        # Scale features
        if fit_scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=numeric_features,
                index=X.index
            )
            self.feature_names = numeric_features
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=numeric_features,
                index=X.index
            )
        
        logger.info(f"Stage 5 complete: {len(numeric_features)} features selected and scaled")
        return X_scaled, numeric_features
    
    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract target variable (Application Review Score)
        """
        return df['Application Review Score']
    
    def get_interview_target(self, df: pd.DataFrame, threshold: int = 19) -> pd.Series:
        """
        Create binary interview target (score >= threshold)
        """
        return (df['Application Review Score'] >= threshold).astype(int)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataTransformationPipeline()
    
    # Transform 2022 data
    df_2022 = pipeline.stage1_load_and_merge("data", 2022)
    df_2022 = pipeline.stage2_handle_missing(df_2022)
    df_2022 = pipeline.stage3_create_ratios(df_2022)
    df_2022 = pipeline.stage4_encode_categoricals(df_2022, fit=True)
    
    # Final scaling (would include LLM scores in practice)
    X_2022, feature_names = pipeline.stage5_select_and_scale(df_2022, fit_scaler=True)
    y_2022 = pipeline.get_target(df_2022)
    
    print(f"Final shape: {X_2022.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Target mean: {y_2022.mean():.2f}")