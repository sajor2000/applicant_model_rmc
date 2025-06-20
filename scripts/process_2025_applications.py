#!/usr/bin/env python3
"""
Production Pipeline for Rush Medical College AI Admissions System
Process 2025 Applications with Essay Analysis and Score Prediction

Author: Juan C. Rojas, MD, MS
Contact: [Contact Email Removed]
Last Updated: December 2024
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineer import FeatureEngineer
from github_repo.src.features.essay_analyzer import EssayAnalyzer
from github_repo.src.processors.application_processor import ApplicationProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """Main production pipeline for processing 2025 applications."""
    
    def __init__(self, model_path='models/cascade_model_2024.pkl'):
        """Initialize the production pipeline."""
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        self.essay_analyzer = EssayAnalyzer()
        self.processor = ApplicationProcessor()
        self.model = None
        
    def load_model(self):
        """Load the trained cascade model."""
        logger.info(f"Loading model from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully")
        
    def process_applications(self, input_file, output_file, essay_dir=None):
        """
        Process applications from Excel file.
        
        Args:
            input_file: Path to "1. Applicants.xlsx"
            output_file: Path for results Excel file
            essay_dir: Directory containing essay PDFs (optional)
        """
        logger.info(f"Starting processing of {input_file}")
        
        # Load applicant data
        logger.info("Loading applicant data...")
        df = pd.read_excel(input_file)
        logger.info(f"Loaded {len(df)} applicants")
        
        # Process essays if directory provided
        if essay_dir:
            logger.info(f"Processing essays from {essay_dir}")
            essay_scores = self.essay_analyzer.analyze_batch(df, essay_dir)
            df = pd.merge(df, essay_scores, on='AMCAS_ID', how='left')
        else:
            logger.warning("No essay directory provided - using mock essay scores")
            # Add mock essay scores for testing
            df = self._add_mock_essay_scores(df)
        
        # Feature engineering
        logger.info("Engineering features...")
        X = self.feature_engineer.transform(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Calculate confidence scores
        confidence_scores = np.max(probabilities, axis=1) * 100
        
        # Prepare results
        results_df = self._prepare_results(df, predictions, confidence_scores)
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        results_df.to_excel(output_file, index=False)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _add_mock_essay_scores(self, df):
        """Add mock essay scores for testing without GPT-4o."""
        essay_features = [
            'llm_overall_essay_score', 'llm_motivation_authenticity',
            'llm_clinical_insight', 'llm_leadership_impact',
            'llm_service_genuineness', 'llm_intellectual_curiosity',
            'llm_maturity_score', 'llm_communication_score',
            'llm_diversity_contribution', 'llm_resilience_score',
            'llm_ethical_reasoning', 'llm_red_flag_count', 'llm_green_flag_count'
        ]
        
        # Generate reasonable mock scores
        np.random.seed(42)
        for feature in essay_features:
            if 'flag' in feature:
                df[feature] = np.random.poisson(1, len(df))
            else:
                df[feature] = np.random.normal(75, 10, len(df))
                df[feature] = np.clip(df[feature], 0, 100)
        
        return df
    
    def _prepare_results(self, df, predictions, confidence_scores):
        """Prepare results dataframe."""
        results = pd.DataFrame({
            'AMCAS_ID': df['AMCAS_ID'] if 'AMCAS_ID' in df else df['Amcas_ID'],
            'Name': df.get('Name', 'N/A'),
            'Predicted_Quartile': predictions,
            'Quartile_Label': pd.Series(predictions).map({
                0: 'Q4 (Bottom 25%)',
                1: 'Q3 (51-75%)',
                2: 'Q2 (26-50%)',
                3: 'Q1 (Top 25%)'
            }),
            'Confidence_Score': confidence_scores,
            'Predicted_Score': self._quartile_to_score(predictions),
            'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Add essay scores if available
        essay_cols = [col for col in df.columns if 'llm_' in col]
        if essay_cols:
            for col in essay_cols:
                results[col] = df[col]
        
        return results
    
    def _quartile_to_score(self, quartiles):
        """Convert quartile predictions to estimated scores."""
        score_map = {
            0: 5,   # Q4: average score ~5
            1: 13,  # Q3: average score ~13
            2: 19,  # Q2: average score ~19
            3: 24   # Q1: average score ~24
        }
        return [score_map[q] for q in quartiles]
    
    def _print_summary(self, results_df):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("PROCESSING COMPLETE - SUMMARY STATISTICS")
        print("="*60)
        print(f"Total Applications Processed: {len(results_df)}")
        print("\nQuartile Distribution:")
        print(results_df['Quartile_Label'].value_counts().sort_index())
        print(f"\nAverage Confidence Score: {results_df['Confidence_Score'].mean():.1f}%")
        print(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description='Process 2025 Rush Medical College Applications'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input Excel file (1. Applicants.xlsx)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results_2025.xlsx',
        help='Path for output Excel file (default: results_2025.xlsx)'
    )
    parser.add_argument(
        '--essays', '-e',
        help='Directory containing essay PDFs (optional)'
    )
    parser.add_argument(
        '--model', '-m',
        default='models/cascade_model_2024.pkl',
        help='Path to trained model file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionPipeline(model_path=args.model)
    
    # Load model
    pipeline.load_model()
    
    # Process applications
    pipeline.process_applications(
        input_file=args.input,
        output_file=args.output,
        essay_dir=args.essays
    )


if __name__ == '__main__':
    main()