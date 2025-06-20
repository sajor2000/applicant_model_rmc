"""
Step 3: Batch Processing Pipeline with Azure OpenAI
==================================================

This script processes all applicants through Azure OpenAI in batches,
with progress tracking, error handling, and cost management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our previous modules
from step1_azure_connection import AzureOpenAIConnection
from step2_extract_unstructured_data import UnstructuredDataExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# System prompt for medical admissions evaluation
SYSTEM_PROMPT = """You are an expert medical school admissions reviewer with 20+ years of experience. Your task is to evaluate applicant essays and unstructured text to predict how human reviewers would score this applicant on a 0-25 scale.

CRITICAL: You must output ONLY numeric scores that can be used as features in a machine learning model. Return a JSON object with exactly these numeric fields:

All scores use 0-10 scale unless specified:
- llm_narrative_coherence: How well the story flows (0-10)
- llm_motivation_authenticity: Genuine vs generic motivation (0-10)
- llm_reflection_depth: Surface vs deep insight (0-10)
- llm_growth_demonstrated: Personal development shown (0-10)
- llm_unique_perspective: What makes them distinctive (0-10)
- llm_clinical_insight: Understanding of healthcare (0-10)
- llm_service_genuineness: Authentic commitment to service (0-10)
- llm_leadership_impact: Real impact created (0-10)
- llm_communication_quality: Writing effectiveness (0-10)
- llm_maturity_score: Professional and emotional maturity (0-10)
- llm_red_flag_count: Count of concerning elements (integer 0-10)
- llm_green_flag_count: Count of exceptional elements (integer 0-10)
- llm_overall_essay_score: Overall quality (0-100)

Focus on substance over style. Value authentic voice. Recognize diverse paths. Consider growth trajectory. Assess readiness for medicine."""


# User prompt template
USER_PROMPT_TEMPLATE = """Please evaluate the following medical school application materials:

PERSONAL STATEMENT:
{personal_statement}

SECONDARY ESSAYS:
{secondary_essays}

EXPERIENCE DESCRIPTIONS:
{experiences}

Based on these materials, provide your numeric evaluation scores in the required JSON format. Remember to use the full scale (0-10) and be consistent in your scoring."""


class BatchProcessor:
    """
    Processes multiple applicants through Azure OpenAI with batch management
    """
    
    def __init__(self, 
                 azure_connection: AzureOpenAIConnection,
                 batch_size: int = 10,
                 delay_between_batches: float = 1.0):
        """
        Initialize batch processor
        
        Args:
            azure_connection: Configured Azure OpenAI connection
            batch_size: Number of applicants per batch
            delay_between_batches: Seconds to wait between batches (rate limiting)
        """
        self.azure = azure_connection
        self.batch_size = batch_size
        self.delay = delay_between_batches
        
        # Track processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'retry_count': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_all_applicants(self, 
                             applicant_data: Dict[str, Dict],
                             resume_from: str = None) -> pd.DataFrame:
        """
        Process all applicants and return DataFrame with LLM scores
        
        Args:
            applicant_data: Dictionary from UnstructuredDataExtractor
            resume_from: AMCAS_ID to resume from (for interrupted processing)
            
        Returns:
            DataFrame with AMCAS_ID and all LLM scores
        """
        self.stats['start_time'] = datetime.now()
        logger.info(f"Starting batch processing for {len(applicant_data)} applicants")
        
        # Convert to list for batching
        applicant_list = list(applicant_data.items())
        
        # Resume from specific ID if requested
        if resume_from:
            resume_idx = next((i for i, (aid, _) in enumerate(applicant_list) 
                             if aid == resume_from), 0)
            applicant_list = applicant_list[resume_idx:]
            logger.info(f"Resuming from {resume_from} ({len(applicant_list)} remaining)")
        
        # Process in batches
        all_results = []
        
        with tqdm(total=len(applicant_list), desc="Processing applicants") as pbar:
            for i in range(0, len(applicant_list), self.batch_size):
                batch = applicant_list[i:i + self.batch_size]
                batch_results = self._process_batch(batch)
                all_results.extend(batch_results)
                
                pbar.update(len(batch))
                
                # Rate limiting
                if i + self.batch_size < len(applicant_list):
                    time.sleep(self.delay)
        
        self.stats['end_time'] = datetime.now()
        self._print_statistics()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Clean up AMCAS_ID (remove year prefix)
        df['AMCAS_ID_original'] = df['AMCAS_ID']
        df['AMCAS_ID'] = df['AMCAS_ID'].str.split('_', n=1).str[1]
        df['year'] = df['AMCAS_ID_original'].str.split('_', n=1).str[0].astype(int)
        
        return df
    
    def _process_batch(self, batch: List[Tuple[str, Dict]]) -> List[Dict]:
        """Process a single batch of applicants"""
        batch_results = []
        
        for amcas_id, content in batch:
            try:
                # Extract content with defaults
                personal_statement = content.get('personal_statement', '')
                secondary_essays = content.get('secondary_essays', {})
                experiences = content.get('experiences', '')
                
                # Skip if no content
                if not any([personal_statement, secondary_essays, experiences]):
                    logger.warning(f"No content for {amcas_id}, skipping")
                    continue
                
                # Get LLM scores
                scores = self.azure.evaluate_essays(
                    personal_statement=personal_statement,
                    secondary_essays=secondary_essays,
                    experiences=experiences,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt_template=USER_PROMPT_TEMPLATE
                )
                
                # Add AMCAS_ID to results
                scores['AMCAS_ID'] = amcas_id
                batch_results.append(scores)
                
                self.stats['successful'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {amcas_id}: {e}")
                self.stats['failed'] += 1
                
                # Add default scores for failed processing
                default_scores = self._get_default_scores()
                default_scores['AMCAS_ID'] = amcas_id
                default_scores['processing_error'] = str(e)
                batch_results.append(default_scores)
        
        self.stats['total_processed'] += len(batch)
        return batch_results
    
    def _get_default_scores(self) -> Dict:
        """Return neutral scores for failed processing"""
        return {
            'llm_narrative_coherence': 5.0,
            'llm_motivation_authenticity': 5.0,
            'llm_reflection_depth': 5.0,
            'llm_growth_demonstrated': 5.0,
            'llm_unique_perspective': 5.0,
            'llm_clinical_insight': 5.0,
            'llm_service_genuineness': 5.0,
            'llm_leadership_impact': 5.0,
            'llm_communication_quality': 5.0,
            'llm_maturity_score': 5.0,
            'llm_red_flag_count': 0,
            'llm_green_flag_count': 0,
            'llm_overall_essay_score': 50.0
        }
    
    def _print_statistics(self):
        """Print processing statistics"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETE")
        print("="*50)
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful']/max(self.stats['total_processed'], 1)*100:.1f}%")
        print(f"Total time: {duration/60:.1f} minutes")
        print(f"Average time per applicant: {duration/max(self.stats['total_processed'], 1):.1f} seconds")
        
        # Estimate costs (rough approximation)
        # Assuming ~2000 tokens input, 100 tokens output per applicant
        estimated_cost = self.stats['successful'] * 0.04  # ~$0.04 per applicant for GPT-4
        print(f"\nEstimated API cost: ${estimated_cost:.2f}")


def save_llm_scores(df: pd.DataFrame, output_path: str = "llm_scores_2022_2023.csv"):
    """
    Save LLM scores in format ready for ML pipeline
    
    Args:
        df: DataFrame with LLM scores
        output_path: Where to save the file
    """
    # Ensure all score columns are float
    score_columns = [col for col in df.columns if col.startswith('llm_')]
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved LLM scores to {output_path}")
    
    # Also save summary statistics
    summary = df[score_columns].describe()
    summary.to_csv(output_path.replace('.csv', '_summary.csv'))
    
    # Print summary
    print("\nLLM Score Summary:")
    print(summary)


# Main execution function
def main():
    """
    Main function to run the complete pipeline
    """
    print("Medical Admissions LLM Scoring Pipeline")
    print("="*50)
    
    # Step 1: Initialize Azure connection
    print("\n1. Initializing Azure OpenAI connection...")
    try:
        azure = AzureOpenAIConnection()
        if not azure.test_connection():
            print("❌ Azure connection failed. Please check credentials.")
            return
        print("✅ Azure connection successful")
    except Exception as e:
        print(f"❌ Failed to initialize Azure: {e}")
        return
    
    # Step 2: Extract unstructured data
    print("\n2. Extracting unstructured data from 2022-2023...")
    extractor = UnstructuredDataExtractor("data")
    applicant_data = extractor.extract_multiple_years([2022, 2023])
    
    stats = extractor.get_summary_statistics(applicant_data)
    print(f"✅ Extracted data for {stats['total_applicants']} applicants")
    print(f"   - Complete data: {stats['complete_data']}")
    
    # Step 3: Process through Azure OpenAI
    print("\n3. Processing applicants through Azure OpenAI...")
    print(f"   This will process {len(applicant_data)} applicants")
    print(f"   Estimated time: {len(applicant_data) * 2 / 60:.1f} minutes")
    print(f"   Estimated cost: ${len(applicant_data) * 0.04:.2f}")
    
    # Confirm before processing
    response = input("\nProceed with processing? (yes/no): ")
    if response.lower() != 'yes':
        print("Processing cancelled.")
        return
    
    # Process applicants
    processor = BatchProcessor(azure, batch_size=10, delay_between_batches=1.0)
    llm_scores_df = processor.process_all_applicants(applicant_data)
    
    # Step 4: Save results
    print("\n4. Saving results...")
    save_llm_scores(llm_scores_df, "llm_scores_2022_2023.csv")
    
    print("\n✅ Pipeline complete!")
    print(f"   Output saved to: llm_scores_2022_2023.csv")
    print(f"   Ready for ML pipeline integration")


if __name__ == "__main__":
    main()