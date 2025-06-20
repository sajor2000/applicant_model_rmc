#!/usr/bin/env python3
"""
Generate GPT-4o LLM scores for training data.
Uses the proven prompts that achieved 80.8% baseline.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import time
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EssayAnalyzerGPT4o:
    """Essay analyzer using GPT-4o with proven prompts."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        
        # Proven prompts from 80.8% baseline
        self.prompts = {
            'overall_essay_score': "Rate the overall quality of this medical school application essay on a scale of 0-100. Focus on clarity, coherence, and compelling narrative. Respond with only the number.",
            'motivation_authenticity': "Rate from 0-100 how genuine and authentic the applicant's motivations for pursuing medicine appear. Look for personal experiences and sincere reflection. Respond with only the number.",
            'clinical_insight': "Rate from 0-100 the depth of clinical understanding and patient care insights demonstrated. Consider both direct experiences and reflective learning. Respond with only the number.",
            'leadership_impact': "Rate from 0-100 the evidence of leadership abilities and tangible positive impacts on others or organizations. Respond with only the number.",
            'service_genuineness': "Rate from 0-100 the genuine commitment to serving others, especially underserved populations. Look for sustained engagement. Respond with only the number.",
            'intellectual_curiosity': "Rate from 0-100 the evidence of intellectual engagement, research aptitude, and growth mindset. Respond with only the number.",
            'maturity_score': "Rate from 0-100 the emotional maturity and professional readiness for medical school. Consider self-awareness and judgment. Respond with only the number.",
            'communication_score': "Rate from 0-100 the effectiveness of written communication, including grammar, style, and narrative flow. Respond with only the number.",
            'diversity_contribution': "Rate from 0-100 the unique perspectives and experiences the applicant would bring to enhance medical school diversity. Respond with only the number.",
            'resilience_score': "Rate from 0-100 the evidence of overcoming challenges and demonstrating resilience. Look for growth through adversity. Respond with only the number.",
            'ethical_reasoning': "Rate from 0-100 the understanding of medical ethics and demonstration of strong personal values. Respond with only the number.",
            'red_flag_count': "Count the number of concerning elements (unprofessionalism, unrealistic expectations, poor judgment). Respond with only the count (0-5).",
            'green_flag_count': "Count the number of exceptional positive indicators (unique achievements, extraordinary insights, exceptional fit). Respond with only the count (0-5)."
        }
    
    def analyze_essay(self, essay_text, applicant_id="unknown"):
        """Analyze an essay and extract all features."""
        features = {}
        
        for feature, prompt in self.prompts.items():
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert medical school admissions evaluator. Provide only numeric scores as requested."},
                        {"role": "user", "content": f"{prompt}\n\nEssay:\n{essay_text[:4000]}"}  # GPT-4o context limit
                    ],
                    temperature=0.15,  # Low temperature for consistency
                    max_tokens=10
                )
                
                value = response.choices[0].message.content.strip()
                
                # Parse response
                try:
                    if 'count' in feature:
                        features[f'llm_{feature}'] = int(float(value))
                    else:
                        features[f'llm_{feature}'] = float(value)
                except ValueError:
                    logger.warning(f"Could not parse response for {feature}: {value}")
                    features[f'llm_{feature}'] = 0 if 'count' in feature else 50.0
                    
            except Exception as e:
                logger.error(f"Error analyzing {feature} for {applicant_id}: {e}")
                features[f'llm_{feature}'] = 0 if 'count' in feature else 50.0
            
            # Rate limiting
            time.sleep(0.5)
        
        return features


def process_year(year, batch_size=10):
    """Process all essays for a given year."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {year} essays with GPT-4o")
    logger.info(f"{'='*60}")
    
    # Load personal statements
    essay_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/9. Personal Statement.xlsx"
    
    if not Path(essay_path).exists():
        logger.error(f"Essay file not found: {essay_path}")
        return None
    
    df = pd.read_excel(essay_path)
    logger.info(f"Loaded {len(df)} essays for {year}")
    
    # Initialize analyzer
    analyzer = EssayAnalyzerGPT4o()
    
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(df), batch_size):
        batch = df.iloc[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        for idx, row in batch.iterrows():
            applicant_id = row.get('amcas_id', row.get('AMCAS ID', idx))
            essay_text = row.get('personal_statement', row.get('Personal Statement', ''))
            
            if pd.isna(essay_text) or not str(essay_text).strip():
                logger.warning(f"Empty essay for {applicant_id}")
                continue
            
            # Analyze essay
            features = analyzer.analyze_essay(str(essay_text), str(applicant_id))
            features['AMCAS ID'] = str(applicant_id)
            results.append(features)
            
            # Progress update
            if len(results) % 20 == 0:
                logger.info(f"  Processed {len(results)} essays...")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} LLM scores to {output_path}")
        return results_df
    
    return None


def main():
    """Generate GPT-4o scores for all training years."""
    logger.info("🚀 GENERATING GPT-4O LLM SCORES")
    logger.info("Target: 2022 and 2023 training data only")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables!")
        return
    
    # Process each year
    for year in ['2022', '2023']:
        # Check if already exists
        output_path = f"data/{year} Applicants Reviewed by Trusted Reviewers/llm_scores_{year}.csv"
        if Path(output_path).exists():
            logger.info(f"LLM scores already exist for {year}, skipping...")
            continue
        
        # Process essays
        df = process_year(year)
        
        if df is not None:
            logger.info(f"✅ Successfully processed {len(df)} essays for {year}")
        else:
            logger.error(f"❌ Failed to process {year}")
    
    logger.info("\n🏁 GPT-4O score generation complete!")


if __name__ == "__main__":
    main()