#!/usr/bin/env python3
"""
Simulate GPT-4o LLM scores for testing the optimization pipeline.
Creates realistic synthetic scores based on application review scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_llm_scores(application_score, seed=None):
    """Generate realistic LLM scores correlated with application review score."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base scores influenced by application score
    # Scores 0-9 (Reject) -> Lower LLM scores
    # Scores 10-15 (Waitlist) -> Medium LLM scores  
    # Scores 16-22 (Interview) -> Higher LLM scores
    # Scores 23-25 (Accept) -> Highest LLM scores
    
    if application_score <= 9:
        base_mean = 45
        std = 15
    elif application_score <= 15:
        base_mean = 60
        std = 12
    elif application_score <= 22:
        base_mean = 75
        std = 10
    else:
        base_mean = 85
        std = 8
    
    # Generate correlated scores with some noise
    scores = {
        'llm_overall_essay_score': np.clip(np.random.normal(base_mean, std), 0, 100),
        'llm_motivation_authenticity': np.clip(np.random.normal(base_mean + 5, std), 0, 100),
        'llm_clinical_insight': np.clip(np.random.normal(base_mean - 5, std * 1.2), 0, 100),
        'llm_leadership_impact': np.clip(np.random.normal(base_mean - 3, std * 1.1), 0, 100),
        'llm_service_genuineness': np.clip(np.random.normal(base_mean + 2, std), 0, 100),
        'llm_intellectual_curiosity': np.clip(np.random.normal(base_mean, std * 0.9), 0, 100),
        'llm_maturity_score': np.clip(np.random.normal(base_mean + 1, std), 0, 100),
        'llm_communication_score': np.clip(np.random.normal(base_mean + 3, std * 0.8), 0, 100),
        'llm_diversity_contribution': np.clip(np.random.normal(base_mean - 2, std * 1.3), 0, 100),
        'llm_resilience_score': np.clip(np.random.normal(base_mean, std * 1.1), 0, 100),
        'llm_ethical_reasoning': np.clip(np.random.normal(base_mean + 4, std * 0.9), 0, 100),
    }
    
    # Flag counts based on score ranges
    if application_score <= 9:
        red_flags = np.random.poisson(1.5)
        green_flags = np.random.poisson(0.3)
    elif application_score <= 15:
        red_flags = np.random.poisson(0.5)
        green_flags = np.random.poisson(0.8)
    elif application_score <= 22:
        red_flags = np.random.poisson(0.2)
        green_flags = np.random.poisson(1.5)
    else:
        red_flags = 0
        green_flags = np.random.poisson(2.5)
    
    scores['llm_red_flag_count'] = min(red_flags, 5)
    scores['llm_green_flag_count'] = min(green_flags, 5)
    
    return scores


def process_year(year):
    """Generate simulated LLM scores for a year."""
    logger.info(f"\nProcessing {year}...")
    
    # Load applicant data
    applicants_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx"
    df = pd.read_excel(applicants_path)
    
    logger.info(f"Loaded {len(df)} applicants")
    
    # Generate LLM scores for each applicant
    llm_scores = []
    for idx, row in df.iterrows():
        # Handle different column names
        if 'AMCAS ID' in df.columns:
            amcas_id = str(row['AMCAS ID'])
        elif 'Amcas_ID' in df.columns:
            amcas_id = str(row['Amcas_ID'])
        else:
            amcas_id = str(idx)
            
        # Handle different column names
        if 'Application Review Score' in df.columns:
            app_score = row['Application Review Score']
        elif 'Application_Review_Score' in df.columns:
            app_score = row['Application_Review_Score']
        else:
            logger.error(f"Could not find application score column. Available: {df.columns.tolist()}")
            app_score = 15  # Default middle score
        
        # Generate scores with some randomness but correlated to review score
        scores = generate_realistic_llm_scores(app_score, seed=int(amcas_id) % 10000)
        scores['AMCAS ID'] = amcas_id
        llm_scores.append(scores)
    
    # Create dataframe
    llm_df = pd.DataFrame(llm_scores)
    
    # Save to file
    output_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv"
    llm_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(llm_df)} LLM scores to {output_path}")
    
    # Show distribution
    logger.info("Score distributions:")
    for col in llm_df.columns:
        if col != 'AMCAS ID' and 'llm_' in col:
            if 'count' in col:
                logger.info(f"  {col}: mean={llm_df[col].mean():.1f}, max={llm_df[col].max()}")
            else:
                logger.info(f"  {col}: mean={llm_df[col].mean():.1f}, std={llm_df[col].std():.1f}")
    
    return llm_df


def main():
    """Generate simulated GPT-4o scores for training data."""
    logger.info("🔧 GENERATING SIMULATED GPT-4O SCORES")
    logger.info("Purpose: Testing optimization pipeline")
    logger.info("Note: These are synthetic scores for development")
    
    for year in ['2022', '2023']:
        # Check if already exists
        output_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv"
        if Path(output_path).exists():
            logger.info(f"\n{year}: LLM scores already exist, skipping...")
            continue
        
        # Generate scores
        process_year(year)
    
    logger.info("\n✅ Simulated scores generated successfully!")
    logger.info("Ready to run optimization pipeline")


if __name__ == "__main__":
    main()