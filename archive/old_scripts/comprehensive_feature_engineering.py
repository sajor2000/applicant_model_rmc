"""
Comprehensive Feature Engineering for Medical Admissions
=======================================================

This module implements the complete feature engineering pipeline combining:
1. Structured data from all Excel files
2. LLM scores from unstructured text analysis
3. Engineered features and interactions
4. Proper encoding and scaling

Target: Application Review Score (0-25)
Interview Threshold: Score ≥ 19 (based on data analysis)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ComprehensiveFeatureEngineer:
    """
    Transforms raw admissions data into ML-ready features optimized
    for predicting Application Review Scores.
    """
    
    def __init__(self):
        # Define interview threshold based on data analysis
        self.INTERVIEW_THRESHOLD = 19  # Scores ≥19 likely get interviews
        
        # Initialize encoders and scalers
        self.one_hot_encoders = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Define feature groups for organized processing
        self.feature_groups = {
            'continuous_academic': [
                'Age',
                'Total_GPA',  # Needs to be derived from Academic Records
                'BCPM_GPA',   # Biology/Chemistry/Physics/Math GPA
                'MCAT_Total', # Needs to be added from MCAT data
                'MCAT_CPBS', 'MCAT_CARS', 'MCAT_BBLS', 'MCAT_PSBB'
            ],
            
            'continuous_experience': [
                'Exp_Hour_Total',
                'Exp_Hour_Research',
                'Exp_Hour_Volunteer_Med',
                'Exp_Hour_Volunteer_Non_Med',
                'Exp_Hour_Employ_Med',
                'Exp_Hour_Shadowing',
                'Comm_Service_Total_Hours',
                'HealthCare_Total_Hours'
            ],
            
            'binary_diversity': [
                'First_Generation_Ind',
                'Disadvantanged_Ind',
                'RU_Ind',  # Rural/Urban
                'Pell_Grant',
                'Fee_Assistance_Program',
                'Childhood_Med_Underserved_Self_Reported',
                'Family_Assistance_Program',
                'Paid_Employment_BF_18'
            ],
            
            'binary_flags': [
                'Prev_Applied_Rush',
                'Inst_Action_Ind',
                'Prev_Matric_Ind',
                'Investigation_Ind',
                'Felony_Ind',
                'Misdemeanor_Ind',
                'Military_Discharge_Ind',
                'Military_HON_Discharge_Ind',
                'Comm_Service_Ind',
                'HealthCare_Ind'
            ],
            
            'categorical_demo': [
                'Gender',
                'Citizenship',
                'SES_Value',
                'Eo_Level',
                'Military_Service'
            ],
            
            'categorical_academic': [
                'Major_Long_Desc',
                'Under_School',
                'Service Rating (Categorical)'
            ],
            
            'ordinal': [
                'Service Rating (Numerical)',  # 1-4 scale
                'Total_GPA_Trend',  # 0/1
                'BCPM_GPA_Trend',   # 0/1
                'Num_Dependents'
            ],
            
            'financial': [
                'Academic_Scholarship_Percentage',
                'Finacial_Need_Based_Percentage',
                'Student_Loan_Percentage',
                'Other_Loan_Percentage',
                'Family_Contribution_Percentage',
                'Applicant_Contribution_Percentage'
            ],
            
            'family_income': [
                'Family_Income_Level',  # 19 categories - needs special handling
                'Number_in_Household'
            ]
        }
        
        # LLM features (from unstructured text analysis)
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
            'llm_red_flag_severity',
            'llm_green_flag_strength',
            'llm_essay_overall_score'
        ]
        
    def load_and_merge_all_data(self, base_path: str, year: int) -> pd.DataFrame:
        """
        Load and merge data from all relevant Excel files
        """
        year_path = f"{base_path}/{year} Applicants Reviewed by Trusted Reviewers"
        
        # Load main applicants file
        df = pd.read_excel(f"{year_path}/1. Applicants.xlsx")
        
        # Standardize ID column
        if 'Amcas_ID' in df.columns:
            df['AMCAS_ID'] = df['Amcas_ID'].astype(str)
        
        # Load and merge language data
        try:
            lang_df = pd.read_excel(f"{year_path}/2. Language.xlsx")
            # Aggregate language features
            lang_features = self._aggregate_language_features(lang_df)
            df = df.merge(lang_features, on='AMCAS_ID', how='left')
        except Exception as e:
            logger.warning(f"Could not load language data: {e}")
        
        # Load and merge parent data
        try:
            parent_df = pd.read_excel(f"{year_path}/3. Parents.xlsx")
            parent_features = self._aggregate_parent_features(parent_df)
            df = df.merge(parent_features, on='AMCAS_ID', how='left')
        except Exception as e:
            logger.warning(f"Could not load parent data: {e}")
        
        # Load and merge sibling data
        try:
            sibling_df = pd.read_excel(f"{year_path}/4. Siblings.xlsx")
            sibling_features = self._aggregate_sibling_features(sibling_df)
            df = df.merge(sibling_features, on='AMCAS_ID', how='left')
        except Exception as e:
            logger.warning(f"Could not load sibling data: {e}")
        
        # Load and merge academic records
        try:
            academic_df = pd.read_excel(f"{year_path}/5. Academic Records.xlsx")
            academic_features = self._aggregate_academic_features(academic_df)
            df = df.merge(academic_features, on='AMCAS_ID', how='left')
        except Exception as e:
            logger.warning(f"Could not load academic data: {e}")
        
        return df
    
    def _aggregate_language_features(self, lang_df: pd.DataFrame) -> pd.DataFrame:
        """Create language diversity features"""
        features = []
        
        for amcas_id, group in lang_df.groupby('AMCAS_ID'):
            feat = {
                'AMCAS_ID': str(amcas_id),
                'num_languages': len(group),
                'has_native_non_english': ((group['Prof_Level'] == 'Native') & 
                                          (group['Language'] != 'English')).any(),
                'num_advanced_languages': (group['Prof_Level'].isin(['Native', 'Advanced'])).sum(),
                'language_diversity_score': len(group) * group['Prof_Level'].map({
                    'Native': 1.0, 'Advanced': 0.8, 'Good': 0.6, 'Fair': 0.4, 'Basic': 0.2
                }).mean()
            }
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _aggregate_parent_features(self, parent_df: pd.DataFrame) -> pd.DataFrame:
        """Create parent/family background features"""
        features = []
        
        # Map education levels to numeric
        edu_map = {
            'Less than High School': 1,
            'High School': 2,
            'Some College': 3,
            'Associate': 4,
            'Bachelor': 5,
            'Master': 6,
            'Doctorate': 7,
            'Professional': 7
        }
        
        for amcas_id, group in parent_df.groupby('AMCAS_ID'):
            parent_df_subset = group.copy()
            parent_df_subset['edu_numeric'] = parent_df_subset['Edu_Level'].map(edu_map).fillna(0)
            
            feat = {
                'AMCAS_ID': str(amcas_id),
                'max_parent_education': parent_df_subset['edu_numeric'].max(),
                'both_parents_college': (parent_df_subset['edu_numeric'] >= 5).sum() >= 2,
                'parent_in_healthcare': parent_df_subset['Occupation'].str.contains(
                    'Medical|Doctor|Physician|Nurse|Health', case=False, na=False
                ).any(),
                'single_parent_household': len(parent_df_subset) == 1
            }
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _aggregate_sibling_features(self, sibling_df: pd.DataFrame) -> pd.DataFrame:
        """Create sibling/family size features"""
        features = []
        
        for amcas_id, group in sibling_df.groupby('AMCAS_ID'):
            feat = {
                'AMCAS_ID': str(amcas_id),
                'num_siblings': len(group),
                'family_size': len(group) + 1,  # Include applicant
                'has_older_siblings': (group['Age'] > group['Age'].iloc[0]).any() if len(group) > 0 else False
            }
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _aggregate_academic_features(self, academic_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GPA and course diversity features"""
        features = []
        
        for amcas_id, group in academic_df.groupby('AMCAS_ID'):
            # Calculate various GPAs
            total_credits = group['Credit_Hrs'].sum()
            
            if total_credits > 0:
                # Overall GPA
                total_gpa = (group['Credit_Hrs'] * group['GPA_Points']).sum() / total_credits
                
                # BCPM GPA (Biology, Chemistry, Physics, Math)
                bcpm_mask = group['Course_Classification'].isin(['Biology', 'Chemistry', 'Physics', 'Math'])
                bcpm_credits = group[bcpm_mask]['Credit_Hrs'].sum()
                
                if bcpm_credits > 0:
                    bcpm_gpa = (group[bcpm_mask]['Credit_Hrs'] * group[bcpm_mask]['GPA_Points']).sum() / bcpm_credits
                else:
                    bcpm_gpa = 0
                
                # Course diversity
                unique_subjects = group['Course_Classification'].nunique()
                
                feat = {
                    'AMCAS_ID': str(amcas_id),
                    'calculated_total_gpa': total_gpa,
                    'calculated_bcpm_gpa': bcpm_gpa,
                    'total_credit_hours': total_credits,
                    'bcpm_credit_hours': bcpm_credits,
                    'course_diversity': unique_subjects,
                    'bcpm_percentage': bcpm_credits / total_credits if total_credits > 0 else 0
                }
            else:
                feat = {
                    'AMCAS_ID': str(amcas_id),
                    'calculated_total_gpa': 0,
                    'calculated_bcpm_gpa': 0,
                    'total_credit_hours': 0,
                    'bcpm_credit_hours': 0,
                    'course_diversity': 0,
                    'bcpm_percentage': 0
                }
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def engineer_features(self, df: pd.DataFrame, llm_scores: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create all engineered features including interactions
        """
        df = df.copy()
        
        # Add calculated features
        epsilon = 1e-6
        
        # Experience ratios
        df['research_intensity'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
        df['clinical_intensity'] = ((df['Exp_Hour_Volunteer_Med'] + df['Exp_Hour_Shadowing']) / 
                                  (df['Exp_Hour_Total'] + epsilon))
        df['volunteer_intensity'] = (df['Exp_Hour_Volunteer_Non_Med'] / 
                                   (df['Exp_Hour_Total'] + epsilon))
        df['employment_intensity'] = df['Exp_Hour_Employ_Med'] / (df['Exp_Hour_Total'] + epsilon)
        
        # Experience balance
        df['research_vs_clinical'] = (df['Exp_Hour_Research'] / 
                                     (df['Exp_Hour_Volunteer_Med'] + df['Exp_Hour_Shadowing'] + epsilon))
        
        # Service commitment score
        df['service_commitment_score'] = (df['Service Rating (Numerical)'] * 
                                         np.log(df['Comm_Service_Total_Hours'] + 1))
        
        # Diversity score (composite of multiple factors)
        diversity_factors = ['First_Generation_Ind', 'Disadvantanged_Ind', 'RU_Ind',
                           'Pell_Grant', 'Fee_Assistance_Program', 
                           'Childhood_Med_Underserved_Self_Reported',
                           'Family_Assistance_Program']
        
        for factor in diversity_factors:
            df[f'{factor}_binary'] = df[factor].map({'Yes': 1, 'No': 0}).fillna(0)
        
        df['diversity_score'] = df[[f'{f}_binary' for f in diversity_factors]].sum(axis=1)
        
        # Academic trajectory
        df['gpa_trend_score'] = df['Total_GPA_Trend'] + df['BCPM_GPA_Trend']
        
        # Age-adjusted experience (older applicants expected to have more hours)
        df['age_adjusted_experience'] = df['Exp_Hour_Total'] / (df['Age'] - 18 + epsilon)
        
        # Financial need indicator
        if 'Family_Income_Level' in df.columns:
            # Convert income to numeric scale
            income_map = {
                'Less than $25,000': 1,
                '$25,000 - $29,999': 2,
                '$30,000 - $39,999': 3,
                '$40,000 - $49,999': 4,
                '$50,000 - $59,999': 5,
                '$60,000 - $74,999': 6,
                '$75,000 - $99,999': 7,
                '$100,000 - $124,999': 8,
                '$125,000 - $149,999': 9,
                '$150,000 - $174,999': 10,
                '$175,000 - $199,999': 11,
                '$200,000 - $249,999': 12,
                '$250,000 - $299,999': 13,
                '$300,000 - $349,999': 14,
                '$350,000 - $399,999': 15,
                '$400,000 or more': 16
            }
            df['income_numeric'] = df['Family_Income_Level'].map(income_map).fillna(8)
            df['financial_need_score'] = 17 - df['income_numeric']  # Inverse scale
        
        # If LLM scores are provided, create interaction features
        if llm_scores is not None:
            df = df.merge(llm_scores, on='AMCAS_ID', how='left')
            
            # Academic package strength
            df['academic_package_score'] = (
                df['calculated_total_gpa'] * df['llm_academic_readiness'] / 10
            )
            
            # Clinical readiness
            df['clinical_readiness_score'] = (
                df['clinical_intensity'] * df['llm_clinical_insight']
            )
            
            # Service authenticity
            df['service_authenticity_score'] = (
                df['service_commitment_score'] * df['llm_service_genuineness'] / 10
            )
            
            # Overall holistic score
            df['holistic_score'] = (
                df['academic_package_score'] * 0.3 +
                df['clinical_readiness_score'] * 0.25 +
                df['service_authenticity_score'] * 0.2 +
                df['diversity_score'] * 0.15 +
                df['llm_essay_overall_score'] * 0.1
            )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode categorical variables
        """
        encoded_dfs = [df]
        
        # Binary encoding (convert Yes/No to 1/0)
        for col in self.feature_groups['binary_diversity'] + self.feature_groups['binary_flags']:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0)
        
        # One-hot encoding for categorical variables
        categorical_cols = (self.feature_groups['categorical_demo'] + 
                          self.feature_groups['categorical_academic'])
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    # Fit and transform
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]].fillna('Unknown'))
                    self.one_hot_encoders[col] = encoder
                else:
                    # Transform only
                    encoded = self.one_hot_encoders[col].transform(df[[col]].fillna('Unknown'))
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                encoded_dfs.append(encoded_df)
        
        # Special handling for Family Income Level (ordinal)
        if 'Family_Income_Level' in df.columns:
            # Already converted to numeric in engineer_features
            pass
        
        # Combine all encoded features
        result = pd.concat(encoded_dfs, axis=1)
        
        # Drop original categorical columns
        cols_to_drop = [col for col in categorical_cols if col in result.columns]
        result = result.drop(columns=cols_to_drop)
        
        return result
    
    def prepare_final_features(self, df: pd.DataFrame, 
                             llm_scores: pd.DataFrame = None,
                             fit_scaler: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare final feature matrix for modeling
        """
        # Engineer features
        df = self.engineer_features(df, llm_scores)
        
        # Encode categorical variables
        df = self.encode_categorical_features(df, fit=fit_scaler)
        
        # Select final features (exclude target and IDs)
        exclude_cols = ['AMCAS_ID', 'Amcas_ID', 'Application Review Score', 
                       'Name', 'Email', 'Phone']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        
        # Scale features
        if fit_scaler:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self.feature_names = feature_cols
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df[feature_cols], feature_cols
    
    def create_target_variables(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create various target variables from Application Review Score
        """
        targets = {}
        
        # Continuous target (0-25)
        targets['score_continuous'] = df['Application Review Score']
        
        # Binary target (Interview: Yes/No)
        targets['interview_binary'] = (df['Application Review Score'] >= self.INTERVIEW_THRESHOLD).astype(int)
        
        # 4-tier classification
        targets['tier_4class'] = pd.cut(
            df['Application Review Score'],
            bins=[-1, 14, 18, 22, 25],
            labels=[0, 1, 2, 3]  # 0: Very Unlikely, 1: Potential, 2: Probable, 3: Very Likely
        )
        
        # Quartile classification
        targets['quartile'] = pd.qcut(
            df['Application Review Score'],
            q=4,
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        return targets


if __name__ == "__main__":
    # Example usage
    engineer = ComprehensiveFeatureEngineer()
    
    # Load 2022 data
    df_2022 = engineer.load_and_merge_all_data("data", 2022)
    
    # Create targets
    targets = engineer.create_target_variables(df_2022)
    
    print(f"Data shape: {df_2022.shape}")
    print(f"Interview rate (≥19): {targets['interview_binary'].mean():.1%}")
    print(f"\nScore distribution:")
    print(df_2022['Application Review Score'].describe())