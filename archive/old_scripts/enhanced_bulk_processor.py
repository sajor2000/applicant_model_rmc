"""
Enhanced Bulk Processing System for Medical Admissions
=====================================================

This module provides an optimized bulk processing pipeline that:
1. Uses the model trained on 2022-2023 data
2. Processes applicants in efficient batches
3. Integrates both structured data and essay analysis
4. Provides real-time progress updates
5. Exports results in multiple formats

Designed specifically for Medical College bulk extraction needs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json

# For essay processing
import PyPDF2
import openai
from sentence_transformers import SentenceTransformer

# For progress tracking
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EssayProcessor:
    """Handles essay extraction and analysis"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_llm = True
        else:
            self.use_llm = False
            logger.warning("No OpenAI API key provided. Essay insights will be limited.")
        
        # Load sentence transformer for essay embeddings
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_essay_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def analyze_essay_with_llm(self, essay_text: str, max_retries: int = 3) -> Dict:
        """Analyze essay using OpenAI GPT-4"""
        if not self.use_llm or not essay_text:
            return self.get_default_essay_scores()
        
        prompt = """
        Analyze this medical school application essay and provide scores (1-10) for:
        1. motivation_score: Clarity and strength of motivation for medicine
        2. experience_quality: Depth and relevance of experiences described
        3. personal_growth: Evidence of maturity and self-reflection
        4. communication_score: Writing quality and storytelling ability
        5. unique_perspective: Distinctive qualities or perspectives
        
        Also provide:
        - key_strengths: List of 3 main strengths (as array)
        - areas_concern: Any red flags or concerns (as array)
        - overall_essay_score: Overall score (1-100)
        
        Return as JSON only.
        
        Essay:
        {essay_text[:4000]}
        """
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert medical school admissions evaluator. Return JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=500
                )
                
                return json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logger.warning(f"LLM analysis attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return self.get_default_essay_scores()
        
        return self.get_default_essay_scores()
    
    def get_default_essay_scores(self) -> Dict:
        """Return default essay scores when LLM is unavailable"""
        return {
            'motivation_score': 5,
            'experience_quality': 5,
            'personal_growth': 5,
            'communication_score': 5,
            'unique_perspective': 5,
            'key_strengths': ['Unable to analyze'],
            'areas_concern': ['Essay not analyzed'],
            'overall_essay_score': 50
        }
    
    def extract_essay_features(self, essay_text: str) -> np.ndarray:
        """Extract numerical features from essay"""
        if not essay_text:
            return np.zeros(390)  # 384 embedding dims + 6 linguistic features
        
        # Get semantic embedding
        embedding = self.sentence_encoder.encode(essay_text)
        
        # Calculate linguistic features
        sentences = essay_text.split('.')
        words = essay_text.split()
        
        linguistic_features = [
            len(words),  # Word count
            len(sentences),  # Sentence count
            np.mean([len(s.split()) for s in sentences if s.strip()]),  # Avg sentence length
            len(set(words)) / len(words) if words else 0,  # Vocabulary diversity
            essay_text.count('!') + essay_text.count('?'),  # Emotional punctuation
            len([w for w in words if len(w) > 10]) / len(words) if words else 0  # Complex word ratio
        ]
        
        return np.concatenate([embedding, linguistic_features])


class BulkAdmissionsProcessor:
    """Main bulk processing pipeline"""
    
    def __init__(self, 
                 model_path: str = "models/best_model_2022_2023_train_2024_test.pkl",
                 batch_size: int = 50,
                 n_workers: int = 4,
                 openai_api_key: Optional[str] = None):
        
        # Load the trained model
        logger.info(f"Loading model from {model_path}")
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']
        
        # Initialize processors
        self.essay_processor = EssayProcessor(openai_api_key)
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        # Results storage
        self.results = []
        
    def load_applicant_data(self, year: int) -> pd.DataFrame:
        """Load applicant data for a specific year"""
        base_path = Path("data") / f"{year} Applicants Reviewed by Trusted Reviewers"
        
        # Load main applicant file
        applicants_df = pd.read_excel(base_path / "1. Applicants.xlsx")
        
        # Try to load additional data files and merge
        additional_files = {
            'personal_statement': '9. Personal Statement.xlsx',
            'secondary_app': '10. Secondary Application.xlsx',
            'experiences': '6. Experiences.xlsx',
            'academic': '5. Academic Records.xlsx'
        }
        
        for key, filename in additional_files.items():
            file_path = base_path / filename
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    # Merge on AMCAS ID
                    if 'AMCAS ID' in df.columns or 'Amcas_ID' in df.columns:
                        id_col = 'AMCAS ID' if 'AMCAS ID' in df.columns else 'Amcas_ID'
                        df = df.rename(columns={id_col: 'AMCAS_ID'})
                        applicants_df = applicants_df.merge(
                            df, 
                            left_on='Amcas_ID', 
                            right_on='AMCAS_ID', 
                            how='left',
                            suffixes=('', f'_{key}')
                        )
                        logger.info(f"Merged {key} data")
                except Exception as e:
                    logger.warning(f"Could not merge {filename}: {e}")
        
        return applicants_df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same preprocessing as training"""
        df = df.copy()
        
        # Handle missing values
        numeric_cols = [
            'Age', 'Exp_Hour_Total', 'Exp_Hour_Research', 
            'Exp_Hour_Volunteer_Med', 'Exp_Hour_Shadowing',
            'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Service Rating (Numerical)', 'Num_Dependents'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle GPA trends
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
        
        # Handle categorical features
        binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).map({'Yes': 1, 'yes': 1}).fillna(0).astype(int)
        
        # Create adversity_overcome feature
        if 'Disadvantanged_Ind' in df.columns:
            df['adversity_overcome'] = df['Disadvantanged_Ind'] * df.get('Total_GPA_Trend', 0)
        
        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        return df[self.feature_names]
    
    async def process_batch_async(self, batch_df: pd.DataFrame, essay_data: Dict) -> List[Dict]:
        """Process a batch of applicants asynchronously"""
        results = []
        
        # Preprocess features
        features = self.preprocess_features(batch_df)
        features_scaled = self.scaler.transform(features)
        
        # Get model predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Process each applicant
        tasks = []
        for idx, (_, applicant) in enumerate(batch_df.iterrows()):
            task = self.process_single_applicant(
                applicant, 
                predictions[idx], 
                probabilities[idx],
                essay_data.get(str(applicant.get('Amcas_ID', '')), '')
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        return results
    
    async def process_single_applicant(self, 
                                     applicant: pd.Series, 
                                     prediction: int, 
                                     probabilities: np.ndarray,
                                     essay_text: str) -> Dict:
        """Process a single applicant with all analyses"""
        
        # Map prediction to tier name
        tier_names = ['Very Unlikely', 'Potential Review', 'Probable Interview', 'Very Likely Interview']
        
        # Base result
        result = {
            'AMCAS_ID': applicant.get('Amcas_ID', 'Unknown'),
            'Age': applicant.get('Age', 0),
            'Predicted_Tier': tier_names[prediction],
            'Tier_Number': prediction + 1,
            'Confidence': float(probabilities.max()),
            'Probability_Distribution': {
                tier_names[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'Total_Experience_Hours': applicant.get('Exp_Hour_Total', 0),
            'Research_Hours': applicant.get('Exp_Hour_Research', 0),
            'Clinical_Hours': applicant.get('Exp_Hour_Volunteer_Med', 0) + applicant.get('Exp_Hour_Shadowing', 0),
            'Service_Rating': applicant.get('Service Rating (Numerical)', 0),
            'GPA_Trend': applicant.get('Total_GPA_Trend', 0),
        }
        
        # Add essay analysis if available
        if essay_text:
            essay_analysis = self.essay_processor.analyze_essay_with_llm(essay_text)
            result['Essay_Analysis'] = essay_analysis
            
            # Calculate combined score
            essay_weight = 0.3
            model_weight = 0.7
            
            # Normalize essay score to 0-3 scale (matching tier numbers)
            normalized_essay_score = (essay_analysis['overall_essay_score'] / 100) * 3
            
            # Combined score
            combined_score = (model_weight * prediction) + (essay_weight * normalized_essay_score)
            result['Combined_Score'] = float(combined_score)
            result['Combined_Tier'] = tier_names[int(round(combined_score))]
        else:
            result['Essay_Analysis'] = None
            result['Combined_Score'] = float(prediction)
            result['Combined_Tier'] = tier_names[prediction]
        
        # Add timestamp
        result['Processed_At'] = datetime.now().isoformat()
        
        return result
    
    async def process_year_async(self, year: int) -> List[Dict]:
        """Process all applicants from a specific year"""
        logger.info(f"\nProcessing {year} applicants...")
        
        # Load data
        df = self.load_applicant_data(year)
        logger.info(f"Loaded {len(df)} applicants from {year}")
        
        # Load essays if available
        essay_data = {}
        essay_path = Path("data") / f"{year} Applicants Reviewed by Trusted Reviewers" / "essays"
        if essay_path.exists():
            for pdf_file in essay_path.glob("*.pdf"):
                amcas_id = pdf_file.stem
                essay_text = self.essay_processor.extract_essay_from_pdf(pdf_file)
                essay_data[amcas_id] = essay_text
            logger.info(f"Loaded {len(essay_data)} essays")
        
        # Process in batches
        all_results = []
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(df), desc=f"Processing {year}") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                batch_results = await self.process_batch_async(batch, essay_data)
                all_results.extend(batch_results)
                pbar.update(len(batch))
        
        return all_results
    
    def export_results(self, results: List[Dict], output_path: str = "admissions_results"):
        """Export results in multiple formats"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by combined score (best candidates first)
        df = df.sort_values('Combined_Score', ascending=False)
        
        # Export formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Excel with multiple sheets
        excel_path = output_path / f"admissions_results_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            df.to_excel(writer, sheet_name='All_Applicants', index=False)
            
            # Per-tier sheets
            for tier in ['Very Likely Interview', 'Probable Interview', 'Potential Review', 'Very Unlikely']:
                tier_df = df[df['Predicted_Tier'] == tier]
                if not tier_df.empty:
                    tier_df.to_excel(writer, sheet_name=tier.replace(' ', '_'), index=False)
            
            # High confidence sheet (confidence > 0.8)
            high_conf = df[df['Confidence'] > 0.8]
            high_conf.to_excel(writer, sheet_name='High_Confidence', index=False)
        
        logger.info(f"Excel results saved to: {excel_path}")
        
        # 2. CSV for easy import
        csv_path = output_path / f"admissions_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV results saved to: {csv_path}")
        
        # 3. Summary statistics
        summary_path = output_path / f"summary_stats_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("ADMISSIONS PROCESSING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Applicants Processed: {len(df)}\n\n")
            
            f.write("Tier Distribution:\n")
            tier_counts = df['Predicted_Tier'].value_counts()
            for tier, count in tier_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"  {tier}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nAverage Confidence: {df['Confidence'].mean():.3f}\n")
            f.write(f"High Confidence Predictions (>80%): {len(df[df['Confidence'] > 0.8])}\n")
            
            if 'Essay_Analysis' in df.columns and df['Essay_Analysis'].notna().any():
                f.write(f"\nEssays Analyzed: {df['Essay_Analysis'].notna().sum()}\n")
        
        logger.info(f"Summary saved to: {summary_path}")
        
        return excel_path, csv_path, summary_path
    
    async def run_bulk_processing(self, years: List[int] = [2024]) -> Tuple[List[Dict], Path]:
        """Main entry point for bulk processing"""
        all_results = []
        
        for year in years:
            year_results = await self.process_year_async(year)
            all_results.extend(year_results)
        
        # Export results
        excel_path, csv_path, summary_path = self.export_results(all_results)
        
        logger.info(f"\nProcessing complete! Total applicants: {len(all_results)}")
        logger.info(f"Results exported to: {excel_path.parent}")
        
        return all_results, excel_path


def main():
    """Example usage"""
    import asyncio
    
    # Initialize processor
    processor = BulkAdmissionsProcessor(
        model_path="models/best_model_2022_2023_train_2024_test.pkl",
        batch_size=50,
        n_workers=4,
        openai_api_key=None  # Add your API key here for essay analysis
    )
    
    # Process 2024 applicants
    results, output_path = asyncio.run(
        processor.run_bulk_processing(years=[2024])
    )
    
    print(f"\nProcessing complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()