"""
Enhanced Feature Engineering for 4-Bucket Classification
======================================================

Creates bucket-aware features optimized for ordinal classification.
Includes polynomial features, interaction terms, and bucket-specific indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    Advanced feature engineering for medical admissions with bucket-aware transformations.
    """
    
    def __init__(self, bucket_boundaries: List[float] = [0, 10, 16, 22, 26]):
        self.bucket_boundaries = bucket_boundaries
        
        # Feature transformers
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        self.pca_high = PCA(n_components=5)  # For high achievers
        self.pca_general = PCA(n_components=10)  # General features
        
        # Feature groups
        self.academic_features = ['Total_GPA', 'BCPM_GPA', 'gpa_difference', 'Total_GPA_Trend', 'BCPM_GPA_Trend']
        self.experience_features = ['Exp_Hour_Total', 'Exp_Hour_Research', 'Exp_Hour_Clinical', 
                                   'Exp_Hour_Volunteer_Med', 'Exp_Hour_Volunteer_Non_Med']
        self.llm_features = ['llm_narrative_coherence', 'llm_motivation_authenticity',
                            'llm_reflection_depth', 'llm_clinical_insight', 'llm_overall_essay_score']
        
        # Fitted parameters
        self.feature_names_ = None
        self.fitted_ = False
        
    def create_bucket_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that indicate potential for each bucket.
        """
        bucket_features = pd.DataFrame(index=df.index)
        
        # Helper function to safely get column
        def safe_get(col, default=0):
            if col in df.columns:
                return df[col].fillna(default)
            else:
                return pd.Series([default]*len(df), index=df.index)
        
        # Reject bucket indicators (0-9)
        bucket_features['low_performance_flags'] = (
            (safe_get('Felony_Ind') == 1).astype(int) +
            (safe_get('Misdemeanor_Ind') == 1).astype(int) +
            (safe_get('Inst_Action_Ind') == 1).astype(int) +
            (safe_get('Investigation_Ind') == 1).astype(int)
        )
        
        # Waitlist bucket indicators (11-15)
        bucket_features['moderate_achievement'] = (
            (safe_get('Exp_Hour_Total') > 500).astype(int) +
            (safe_get('First_Generation_Ind') == 1).astype(int) +
            (safe_get('Disadvantanged_Ind') == 1).astype(int)
        )
        
        # Interview bucket indicators (17-21)
        bucket_features['strong_clinical'] = (
            (safe_get('Exp_Hour_Clinical') > 200).astype(int) +
            (safe_get('Exp_Hour_Shadowing') > 100).astype(int) +
            (safe_get('HealthCare_Total_Hours') > 500).astype(int)
        )
        
        # Accept bucket indicators (23-25)
        bucket_features['exceptional_markers'] = (
            (safe_get('Exp_Hour_Research') > 1000).astype(int) +
            (safe_get('llm_overall_essay_score', 70) > 85).astype(int) +
            (safe_get('Comm_Service_Total_Hours') > 1000).astype(int)
        )
        
        return bucket_features
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create meaningful ratio features.
        """
        ratio_features = pd.DataFrame(index=df.index)
        epsilon = 1e-8
        
        # Helper function
        def safe_get(col, default=0):
            if col in df.columns:
                return df[col].fillna(default)
            else:
                return pd.Series([default]*len(df), index=df.index)
        
        # Experience ratios
        total_exp = safe_get('Exp_Hour_Total') + epsilon
        ratio_features['research_ratio'] = safe_get('Exp_Hour_Research') / total_exp
        ratio_features['clinical_ratio'] = safe_get('Exp_Hour_Clinical') / total_exp
        ratio_features['volunteer_ratio'] = (safe_get('Exp_Hour_Volunteer_Med') + 
                                            safe_get('Exp_Hour_Volunteer_Non_Med')) / total_exp
        
        # Balance metrics
        ratio_features['experience_diversity'] = 1 - (
            ratio_features['research_ratio']**2 + 
            ratio_features['clinical_ratio']**2 + 
            ratio_features['volunteer_ratio']**2
        )
        
        # Commitment indicators
        age = safe_get('Age', 22)
        ratio_features['experience_per_year'] = total_exp / (age - 18 + epsilon)
        
        # LLM consistency
        if 'llm_narrative_coherence' in df.columns:
            llm_cols = [col for col in df.columns if col.startswith('llm_') and 'overall' not in col and 'count' not in col]
            if llm_cols:
                ratio_features['llm_consistency'] = df[llm_cols].std(axis=1)
                ratio_features['llm_strength'] = df[llm_cols].mean(axis=1)
        
        return ratio_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key predictors.
        """
        interaction_features = pd.DataFrame(index=df.index)
        
        # Academic × Experience interactions
        if 'Total_GPA' in df.columns and 'Exp_Hour_Total' in df.columns:
            interaction_features['gpa_x_experience'] = df['Total_GPA'] * np.log1p(df['Exp_Hour_Total'])
        
        # Disadvantage × Achievement interactions
        if 'Disadvantanged_Ind' in df.columns:
            disadvantaged = df['Disadvantanged_Ind']
            if 'Exp_Hour_Total' in df.columns:
                interaction_features['disadvantaged_x_hours'] = disadvantaged * df['Exp_Hour_Total']
            if 'llm_overall_essay_score' in df.columns:
                interaction_features['disadvantaged_x_essay'] = disadvantaged * df['llm_overall_essay_score']
        
        # Service × Leadership
        if 'Comm_Service_Total_Hours' in df.columns and 'llm_leadership_impact' in df.columns:
            interaction_features['service_x_leadership'] = (
                np.log1p(df['Comm_Service_Total_Hours']) * df['llm_leadership_impact']
            )
        
        # Research × Clinical balance
        if 'Exp_Hour_Research' in df.columns and 'Exp_Hour_Clinical' in df.columns:
            research = np.log1p(df['Exp_Hour_Research'])
            clinical = np.log1p(df['Exp_Hour_Clinical'])
            interaction_features['research_clinical_balance'] = research * clinical / (research + clinical + 1)
        
        return interaction_features
    
    def create_polynomial_features(self, df: pd.DataFrame, feature_subset: List[str]) -> pd.DataFrame:
        """
        Create polynomial features for key numeric predictors.
        """
        # Select available features
        available_features = [f for f in feature_subset if f in df.columns]
        if not available_features:
            return pd.DataFrame(index=df.index)
        
        # Extract subset
        X_subset = df[available_features].fillna(0)
        
        # Create polynomial features
        X_poly = self.poly_transformer.fit_transform(X_subset)
        
        # Get feature names
        poly_names = []
        for i, name1 in enumerate(available_features):
            poly_names.append(name1)  # Linear terms
            for j in range(i, len(available_features)):
                name2 = available_features[j]
                if i == j:
                    poly_names.append(f"{name1}^2")
                else:
                    poly_names.append(f"{name1}_x_{name2}")
        
        # Create DataFrame
        poly_df = pd.DataFrame(X_poly, columns=poly_names[:X_poly.shape[1]], index=df.index)
        
        return poly_df
    
    def create_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that indicate proximity to bucket thresholds.
        """
        threshold_features = pd.DataFrame(index=df.index)
        
        # For each important feature, create threshold indicators
        if 'Exp_Hour_Total' in df.columns:
            exp_total = df['Exp_Hour_Total']
            threshold_features['exp_low'] = (exp_total < 500).astype(int)
            threshold_features['exp_moderate'] = ((exp_total >= 500) & (exp_total < 1500)).astype(int)
            threshold_features['exp_high'] = (exp_total >= 1500).astype(int)
        
        if 'llm_overall_essay_score' in df.columns:
            essay = df['llm_overall_essay_score']
            threshold_features['essay_weak'] = (essay < 65).astype(int)
            threshold_features['essay_good'] = ((essay >= 65) & (essay < 80)).astype(int)
            threshold_features['essay_excellent'] = (essay >= 80).astype(int)
        
        # Red flag combinations
        red_flags = []
        for flag in ['Felony_Ind', 'Misdemeanor_Ind', 'Inst_Action_Ind']:
            if flag in df.columns:
                # Convert Yes/No to 1/0
                flag_series = df[flag].copy()
                if flag_series.dtype == 'object':
                    flag_series = flag_series.map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0, 1: 1, 0: 0})
                    flag_series = flag_series.fillna(0)
                red_flags.append(flag_series)
        if red_flags:
            red_flags_df = pd.concat(red_flags, axis=1)
            threshold_features['any_red_flag'] = red_flags_df.max(axis=1)
            threshold_features['multiple_red_flags'] = (red_flags_df.sum(axis=1) > 1).astype(int)
        
        return threshold_features
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Fit and transform features.
        """
        self.fitted_ = True
        
        # Create all feature groups
        features_list = []
        
        # 1. Original features (cleaned)
        # First convert Yes/No columns to numeric
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                unique_vals = df_cleaned[col].dropna().unique()
                if set(unique_vals).issubset({'Yes', 'No', 'Y', 'N'}):
                    df_cleaned[col] = df_cleaned[col].map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0})
                elif set(unique_vals).issubset({'Male', 'Female', 'M', 'F'}):
                    df_cleaned[col] = df_cleaned[col].map({'Male': 1, 'M': 1, 'Female': 0, 'F': 0})
        
        # CRITICAL: Remove target and ID columns
        exclude_cols = ['Application Review Score', 'AMCAS_ID', 'Amcas_ID', 'AMCAS ID', 'year']
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        features_list.append(df_cleaned[numeric_cols])
        
        # 2. Bucket indicators
        bucket_indicators = self.create_bucket_indicators(df_cleaned)
        features_list.append(bucket_indicators)
        
        # 3. Ratio features
        ratio_features = self.create_ratio_features(df_cleaned)
        features_list.append(ratio_features)
        
        # 4. Interaction features
        interaction_features = self.create_interaction_features(df_cleaned)
        features_list.append(interaction_features)
        
        # 5. Polynomial features for top predictors
        key_features = ['Exp_Hour_Total', 'llm_overall_essay_score', 'Total_GPA']
        poly_features = self.create_polynomial_features(df_cleaned, key_features)
        if not poly_features.empty:
            features_list.append(poly_features)
        
        # 6. Threshold features
        threshold_features = self.create_threshold_features(df_cleaned)
        features_list.append(threshold_features)
        
        # Combine all features
        enhanced_df = pd.concat(features_list, axis=1)
        
        # Remove duplicate columns
        enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
        
        # Store feature names
        self.feature_names_ = enhanced_df.columns.tolist()
        
        # Scale features
        enhanced_scaled = pd.DataFrame(
            self.scaler.fit_transform(enhanced_df),
            columns=enhanced_df.columns,
            index=enhanced_df.index
        )
        
        print(f"\n✓ Enhanced features created:")
        print(f"  Original features: {len(numeric_cols)}")
        print(f"  Enhanced features: {len(enhanced_scaled.columns)}")
        print(f"  Feature expansion: {len(enhanced_scaled.columns) / len(numeric_cols):.1f}x")
        
        return enhanced_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted parameters.
        """
        if not self.fitted_:
            raise ValueError("Must fit before transform")
        
        # Apply same transformations as fit_transform
        features_list = []
        
        # Clean data first
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                unique_vals = df_cleaned[col].dropna().unique()
                if set(unique_vals).issubset({'Yes', 'No', 'Y', 'N'}):
                    df_cleaned[col] = df_cleaned[col].map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0})
                elif set(unique_vals).issubset({'Male', 'Female', 'M', 'F'}):
                    df_cleaned[col] = df_cleaned[col].map({'Male': 1, 'M': 1, 'Female': 0, 'F': 0})
        
        # Original features
        exclude_cols = ['Application Review Score', 'AMCAS_ID', 'Amcas_ID', 'AMCAS ID', 'year']
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        features_list.append(df_cleaned[numeric_cols])
        
        # All feature groups
        features_list.append(self.create_bucket_indicators(df_cleaned))
        features_list.append(self.create_ratio_features(df_cleaned))
        features_list.append(self.create_interaction_features(df_cleaned))
        
        key_features = ['Exp_Hour_Total', 'llm_overall_essay_score', 'Total_GPA']
        poly_features = self.create_polynomial_features(df_cleaned, key_features)
        if not poly_features.empty:
            features_list.append(poly_features)
            
        features_list.append(self.create_threshold_features(df_cleaned))
        
        # Combine
        enhanced_df = pd.concat(features_list, axis=1)
        enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
        
        # Ensure same columns as training
        missing_cols = set(self.feature_names_) - set(enhanced_df.columns)
        for col in missing_cols:
            enhanced_df[col] = 0
            
        enhanced_df = enhanced_df[self.feature_names_]
        
        # Scale
        enhanced_scaled = pd.DataFrame(
            self.scaler.transform(enhanced_df),
            columns=enhanced_df.columns,
            index=enhanced_df.index
        )
        
        return enhanced_scaled
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Return feature names organized by group.
        """
        if not self.fitted_:
            return {}
            
        groups = {
            'bucket_indicators': [],
            'ratios': [],
            'interactions': [],
            'polynomials': [],
            'thresholds': [],
            'original': []
        }
        
        for feature in self.feature_names_:
            if any(x in feature for x in ['_flags', '_achievement', '_clinical', '_markers']):
                groups['bucket_indicators'].append(feature)
            elif any(x in feature for x in ['_ratio', '_diversity', '_per_year', '_consistency']):
                groups['ratios'].append(feature)
            elif '_x_' in feature:
                groups['interactions'].append(feature)
            elif '^2' in feature:
                groups['polynomials'].append(feature)
            elif any(x in feature for x in ['_low', '_moderate', '_high', '_weak', '_good', '_excellent']):
                groups['thresholds'].append(feature)
            else:
                groups['original'].append(feature)
                
        return groups


# Example usage
if __name__ == "__main__":
    print("Enhanced Feature Engineering for Medical Admissions")
    print("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Total_GPA': [3.8, 3.5, 3.9, 3.2],
        'Exp_Hour_Total': [1200, 800, 2000, 500],
        'Exp_Hour_Research': [800, 200, 1500, 100],
        'Exp_Hour_Clinical': [200, 400, 300, 300],
        'llm_overall_essay_score': [85, 72, 90, 65],
        'llm_narrative_coherence': [8, 7, 9, 6],
        'Disadvantanged_Ind': [0, 1, 0, 1],
        'Felony_Ind': [0, 0, 0, 1]
    })
    
    # Transform features
    engineer = EnhancedFeatureEngineer()
    enhanced_features = engineer.fit_transform(sample_data)
    
    # Show feature groups
    print("\n✓ Feature groups:")
    groups = engineer.get_feature_groups()
    for group_name, features in groups.items():
        if features:
            print(f"  {group_name}: {len(features)} features")
            print(f"    Examples: {features[:3]}")