"""Feature engineering for medical school applications."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from application data."""
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data.
        
        Args:
            df: DataFrame with raw application data
            
        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()
        
        # Essay-service alignment
        if 'llm_overall_essay_score' in df.columns and 'service_rating_numerical' in df.columns:
            essay_norm = df['llm_overall_essay_score'] / 100
            service_norm = (df['service_rating_numerical'] - 1) / 3
            df['essay_service_alignment'] = 1 - abs(essay_norm - service_norm)
            df['service_essay_product'] = df['service_rating_numerical'] * df['llm_overall_essay_score'] / 25
        
        # Flag balance
        if 'llm_red_flag_count' in df.columns and 'llm_green_flag_count' in df.columns:
            df['flag_balance'] = df['llm_green_flag_count'] - df['llm_red_flag_count']
            df['flag_ratio'] = df['llm_green_flag_count'] / (df['llm_red_flag_count'] + 1)
        
        # Service × Clinical interaction
        if 'service_rating_numerical' in df.columns and 'healthcare_total_hours' in df.columns:
            df['service_clinical_log'] = df['service_rating_numerical'] * np.log1p(df['healthcare_total_hours'])
        
        # Experience features
        exp_cols = ['exp_hour_research', 'exp_hour_volunteer_med', 'exp_hour_volunteer_non_med']
        available_exp = [col for col in exp_cols if col in df.columns]
        
        if len(available_exp) > 1:
            # Experience diversity
            df['experience_diversity'] = (df[available_exp] > 50).sum(axis=1)
            
            # Experience consistency
            exp_mean = df[available_exp].mean(axis=1)
            exp_std = df[available_exp].std(axis=1)
            df['experience_consistency'] = 1 / (1 + exp_std / (exp_mean + 1))
        
        # Profile coherence
        if 'llm_overall_essay_score' in df.columns:
            struct_features = ['healthcare_total_hours', 'exp_hour_research', 
                             'exp_hour_volunteer_med', 'service_rating_numerical']
            essay_features = ['llm_clinical_insight', 'llm_service_genuineness',
                            'llm_motivation_authenticity', 'llm_leadership_impact']
            
            available_struct = [f for f in struct_features if f in df.columns]
            available_essay = [f for f in essay_features if f in df.columns]
            
            if len(available_struct) > 0 and len(available_essay) > 0:
                # Normalize features
                for feat in available_struct + available_essay:
                    if feat in df.columns:
                        df[f'{feat}_norm'] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-6)
                
                # Calculate coherence
                df['profile_coherence'] = 0
                coherence_count = 0
                
                for s_feat in available_struct:
                    for e_feat in available_essay:
                        if f'{s_feat}_norm' in df.columns and f'{e_feat}_norm' in df.columns:
                            df['profile_coherence'] += df[f'{s_feat}_norm'] * df[f'{e_feat}_norm']
                            coherence_count += 1
                
                if coherence_count > 0:
                    df['profile_coherence'] = df['profile_coherence'] / coherence_count
                
                # Drop temporary normalized columns
                temp_cols = [col for col in df.columns if '_norm' in col]
                df = df.drop(columns=temp_cols)
        
        # Clinical readiness score
        clinical_features = ['healthcare_total_hours', 'llm_clinical_insight', 'exp_hour_shadowing']
        available_clinical = [f for f in clinical_features if f in df.columns]
        
        if len(available_clinical) > 0:
            for feat in available_clinical:
                df[f'{feat}_clinical_norm'] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min() + 1e-6)
            
            clinical_norm_cols = [f'{feat}_clinical_norm' for feat in available_clinical]
            df['clinical_readiness_score'] = df[clinical_norm_cols].mean(axis=1)
            
            # Drop temporary columns
            df = df.drop(columns=clinical_norm_cols)
        
        # Academic potential score
        academic_features = ['exp_hour_research', 'llm_intellectual_curiosity', 'llm_maturity_score']
        available_academic = [f for f in academic_features if f in df.columns]
        
        if len(available_academic) > 0:
            for feat in available_academic:
                df[f'{feat}_academic_norm'] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min() + 1e-6)
            
            academic_norm_cols = [f'{feat}_academic_norm' for feat in available_academic]
            df['academic_potential_score'] = df[academic_norm_cols].mean(axis=1)
            
            # Drop temporary columns
            df = df.drop(columns=academic_norm_cols)
        
        logger.info(f"Engineered {len(df.columns) - len(available_exp) - len(available_struct)} new features")
        
        return df