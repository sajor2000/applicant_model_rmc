"""
Integrated Pipeline: Structured Data Model + Azure Essay Scoring
===============================================================

This script combines:
1. The ML model trained on 2022-2023 structured data
2. Azure OpenAI essay scoring for unstructured data
3. A final integrated scoring system

The final score weights both components for holistic evaluation.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from azure_essay_scorer import AzureEssayScorer, create_azure_config
from train_test_2024 import Train2024TestPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedAzureAdmissionsSystem:
    """
    Complete admissions evaluation system combining structured data ML
    and Azure OpenAI essay analysis.
    """
    
    def __init__(self,
                 model_path: str = "models/best_model_2022_2023_train_2024_test.pkl",
                 azure_config: Optional[Dict] = None,
                 essay_weight: float = 0.35,
                 structured_weight: float = 0.65):
        
        # Load trained ML model
        logger.info("Loading structured data model...")
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']
        
        # Initialize Azure essay scorer
        logger.info("Initializing Azure OpenAI essay scorer...")
        if azure_config is None:
            azure_config = create_azure_config()
        self.essay_scorer = AzureEssayScorer(**azure_config)
        
        # Scoring weights
        self.essay_weight = essay_weight
        self.structured_weight = structured_weight
        
        # Tier definitions
        self.tier_names = [
            'Very Unlikely',
            'Potential Review',
            'Probable Interview',
            'Very Likely Interview'
        ]
        
    def load_essays_from_excel(self, excel_path: Path, 
                              id_column: str = 'AMCAS_ID',
                              essay_column: str = 'Personal_Statement_Text') -> Dict[str, str]:
        """Load essays from Excel file"""
        try:
            df = pd.read_excel(excel_path)
            essays = {}
            for _, row in df.iterrows():
                if pd.notna(row.get(essay_column)):
                    essays[str(row[id_column])] = str(row[essay_column])
            logger.info(f"Loaded {len(essays)} essays from {excel_path}")
            return essays
        except Exception as e:
            logger.error(f"Error loading essays from {excel_path}: {e}")
            return {}
    
    def prepare_structured_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare structured features (same as training pipeline)"""
        df = df.copy()
        
        # Numeric columns
        numeric_cols = [
            'Age', 'Exp_Hour_Total', 'Exp_Hour_Research', 
            'Exp_Hour_Volunteer_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Service Rating (Numerical)', 'Num_Dependents'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # GPA trends
        gpa_cols = ['Total_GPA_Trend', 'BCPM_GPA_Trend']
        for col in gpa_cols:
            if col in df.columns:
                df[col] = df[col].replace('NULL', 0).fillna(0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Feature engineering
        epsilon = 1e-6
        df['research_intensity'] = df['Exp_Hour_Research'] / (df['Exp_Hour_Total'] + epsilon)
        df['clinical_intensity'] = (
            (df.get('Exp_Hour_Volunteer_Med', 0) + df.get('Exp_Hour_Shadowing', 0)) / 
            (df['Exp_Hour_Total'] + epsilon)
        )
        df['experience_balance'] = (
            df['Exp_Hour_Research'] / 
            (df.get('Exp_Hour_Volunteer_Med', 0) + df.get('Exp_Hour_Shadowing', 0) + epsilon)
        )
        df['service_commitment'] = (
            df.get('Service Rating (Numerical)', 0) * 
            np.log(df['Comm_Service_Total_Hours'] + 1)
        )
        
        # Binary features
        binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).map({'Yes': 1, 'yes': 1}).fillna(0).astype(int)
        
        if 'Disadvantanged_Ind' in df.columns:
            df['adversity_overcome'] = df['Disadvantanged_Ind'] * df.get('Total_GPA_Trend', 0)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        return df[self.feature_names]
    
    def calculate_integrated_score(self,
                                 structured_prediction: int,
                                 structured_proba: np.ndarray,
                                 essay_scores: Dict) -> Dict:
        """
        Calculate final integrated score combining structured and essay evaluations
        """
        
        # Normalize essay overall score to 0-3 scale (matching tier indices)
        essay_normalized = (essay_scores['overall_score'] / 100) * 3
        
        # Calculate weighted score
        integrated_score = (
            self.structured_weight * structured_prediction +
            self.essay_weight * essay_normalized
        )
        
        # Round to nearest tier
        integrated_tier = int(round(integrated_score))
        integrated_tier = max(0, min(3, integrated_tier))  # Ensure in range
        
        # Calculate confidence based on agreement
        score_difference = abs(structured_prediction - essay_normalized)
        agreement_factor = 1 - (score_difference / 3)  # Normalize to 0-1
        
        # Combine structured confidence with agreement
        structured_confidence = float(structured_proba.max())
        integrated_confidence = (structured_confidence + agreement_factor) / 2
        
        # Identify key factors
        key_factors = []
        
        # From structured data
        if structured_prediction >= 2:  # Interview likely
            key_factors.append("Strong quantitative metrics")
        
        # From essay
        if essay_scores['motivation_score'] >= 8:
            key_factors.append("Exceptional motivation for medicine")
        if essay_scores['clinical_understanding'] >= 8:
            key_factors.append("Deep clinical understanding")
        if essay_scores['leadership_score'] >= 8:
            key_factors.append("Strong leadership potential")
        
        # Red flags consideration
        red_flags = essay_scores.get('red_flags', [])
        if red_flags and integrated_tier >= 2:
            integrated_tier = max(integrated_tier - 1, 1)  # Downgrade but not to lowest
            integrated_confidence *= 0.8
        
        return {
            'integrated_tier': integrated_tier,
            'integrated_tier_name': self.tier_names[integrated_tier],
            'integrated_score': float(integrated_score),
            'integrated_confidence': float(integrated_confidence),
            'structured_tier': structured_prediction,
            'essay_tier_equivalent': float(essay_normalized),
            'agreement_score': float(agreement_factor),
            'key_factors': key_factors,
            'recommendation': essay_scores.get('recommendation', 'Neutral')
        }
    
    async def evaluate_applicant_async(self,
                                     applicant_data: pd.Series,
                                     essay_text: Optional[str] = None) -> Dict:
        """Evaluate a single applicant with both models"""
        
        # Prepare structured features
        features_df = pd.DataFrame([applicant_data])
        features_df = self.prepare_structured_features(features_df)
        features_scaled = self.scaler.transform(features_df)
        
        # Get structured prediction
        structured_pred = self.model.predict(features_scaled)[0]
        structured_proba = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'AMCAS_ID': applicant_data.get('Amcas_ID', 'Unknown'),
            'structured_prediction': int(structured_pred),
            'structured_tier_name': self.tier_names[structured_pred],
            'structured_confidence': float(structured_proba.max()),
            'structured_probabilities': {
                self.tier_names[i]: float(p) for i, p in enumerate(structured_proba)
            }
        }
        
        # If essay provided, get essay scores and integrate
        if essay_text:
            # Prepare context for essay scoring
            context = {
                'age': applicant_data.get('Age', 'Unknown'),
                'total_hours': applicant_data.get('Exp_Hour_Total', 'Unknown'),
                'research_hours': applicant_data.get('Exp_Hour_Research', 'Unknown'),
                'clinical_hours': (
                    applicant_data.get('Exp_Hour_Volunteer_Med', 0) + 
                    applicant_data.get('Exp_Hour_Shadowing', 0)
                ),
                'gpa_trend': applicant_data.get('Total_GPA_Trend', 'Unknown')
            }
            
            # Get essay scores
            essay_scores = await self.essay_scorer.score_essay_async(essay_text, context)
            result['essay_scores'] = essay_scores
            
            # Calculate integrated score
            integration = self.calculate_integrated_score(
                structured_pred, structured_proba, essay_scores
            )
            result.update(integration)
        else:
            # No essay - use structured only
            result['integrated_tier'] = structured_pred
            result['integrated_tier_name'] = self.tier_names[structured_pred]
            result['integrated_score'] = float(structured_pred)
            result['integrated_confidence'] = float(structured_proba.max())
            result['essay_scores'] = None
        
        result['evaluated_at'] = datetime.now().isoformat()
        
        return result
    
    async def evaluate_batch_async(self,
                                 applicants_df: pd.DataFrame,
                                 essays: Dict[str, str],
                                 batch_size: int = 20) -> List[Dict]:
        """Evaluate a batch of applicants"""
        
        results = []
        total = len(applicants_df)
        
        for i in range(0, total, batch_size):
            batch_df = applicants_df.iloc[i:i+batch_size]
            
            # Create evaluation tasks
            tasks = []
            for _, applicant in batch_df.iterrows():
                amcas_id = str(applicant.get('Amcas_ID', ''))
                essay_text = essays.get(amcas_id)
                
                task = self.evaluate_applicant_async(applicant, essay_text)
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            logger.info(f"Processed {min(i + batch_size, total)}/{total} applicants")
        
        return results
    
    def generate_evaluation_report(self, results: List[Dict]) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        
        df = pd.DataFrame(results)
        
        # Add summary statistics
        if 'integrated_tier' in df.columns:
            tier_distribution = df['integrated_tier_name'].value_counts()
            logger.info("\nIntegrated Tier Distribution:")
            for tier, count in tier_distribution.items():
                logger.info(f"  {tier}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sort by integrated score (best first)
        df = df.sort_values('integrated_score', ascending=False)
        
        # Add ranking
        df['rank'] = range(1, len(df) + 1)
        
        # Flag high-confidence recommendations
        df['high_confidence_interview'] = (
            (df['integrated_tier'] >= 2) & 
            (df['integrated_confidence'] > 0.8)
        )
        
        return df
    
    def export_results(self, df: pd.DataFrame, output_dir: str = "integrated_results"):
        """Export results in multiple formats"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Excel with multiple sheets
        excel_path = output_path / f"integrated_evaluation_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # All applicants
            df.to_excel(writer, sheet_name='All_Applicants', index=False)
            
            # Interview recommendations
            interview_df = df[df['integrated_tier'] >= 2].copy()
            interview_df.to_excel(writer, sheet_name='Interview_Candidates', index=False)
            
            # High confidence
            high_conf_df = df[df['integrated_confidence'] > 0.8].copy()
            high_conf_df.to_excel(writer, sheet_name='High_Confidence', index=False)
            
            # Essay insights (if available)
            if 'essay_scores' in df.columns and df['essay_scores'].notna().any():
                essay_df = df[df['essay_scores'].notna()].copy()
                
                # Extract essay scores
                for col in ['motivation_score', 'clinical_understanding', 'service_commitment']:
                    essay_df[col] = essay_df['essay_scores'].apply(
                        lambda x: x.get(col, 0) if isinstance(x, dict) else 0
                    )
                
                essay_summary = essay_df[['AMCAS_ID', 'motivation_score', 
                                         'clinical_understanding', 'service_commitment',
                                         'recommendation']].copy()
                essay_summary.to_excel(writer, sheet_name='Essay_Analysis', index=False)
        
        logger.info(f"Results exported to: {excel_path}")
        
        # CSV for easy import
        csv_path = output_path / f"integrated_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        return excel_path


async def run_integrated_evaluation(year: int = 2024,
                                  azure_config: Optional[Dict] = None):
    """Run complete integrated evaluation pipeline"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"INTEGRATED EVALUATION PIPELINE - {year} Applicants")
    logger.info(f"{'='*60}\n")
    
    # Initialize system
    system = IntegratedAzureAdmissionsSystem(azure_config=azure_config)
    
    # Load applicant data
    data_path = Path("data") / f"{year} Applicants Reviewed by Trusted Reviewers"
    applicants_df = pd.read_excel(data_path / "1. Applicants.xlsx")
    logger.info(f"Loaded {len(applicants_df)} applicants")
    
    # Load essays
    essays = {}
    essay_file = data_path / "9. Personal Statement.xlsx"
    if essay_file.exists():
        essays = system.load_essays_from_excel(essay_file)
    logger.info(f"Loaded {len(essays)} essays")
    
    # Evaluate all applicants
    results = await system.evaluate_batch_async(applicants_df, essays)
    
    # Generate report
    report_df = system.generate_evaluation_report(results)
    
    # Export results
    output_path = system.export_results(report_df)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*60}\n")
    
    return report_df


# Main execution
if __name__ == "__main__":
    # Example Azure configuration
    azure_config = {
        'azure_endpoint': 'https://your-resource.openai.azure.com/',
        'api_key': 'your-api-key',
        'deployment_name': 'gpt-4',
        'api_version': '2024-02-15-preview'
    }
    
    # Run evaluation
    results = asyncio.run(run_integrated_evaluation(
        year=2024,
        azure_config=azure_config
    ))