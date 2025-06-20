#!/usr/bin/env python3
"""Process medical school applications through the AI model."""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processors import ApplicationProcessor


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process medical school applications')
    parser.add_argument('input', help='Input CSV/Excel file with applications')
    parser.add_argument('-o', '--output', help='Output file path', 
                       default='processed_applications.csv')
    parser.add_argument('--model', help='Path to model file', default=None)
    
    args = parser.parse_args()
    
    # Load applications
    print(f"Loading applications from {args.input}...")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)
    
    print(f"Loaded {len(df)} applications")
    
    # Initialize processor
    processor = ApplicationProcessor(model_path=args.model)
    
    # Process applications
    print("Processing applications...")
    results = processor.process_batch(
        df,
        progress_callback=lambda curr, total: print(f"Progress: {curr}/{total}", end='\r')
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    results.to_csv(args.output, index=False)
    
    # Summary statistics
    print("\nSummary:")
    print(f"Total processed: {len(results)}")
    if 'predicted_quartile' in results.columns:
        print("\nQuartile distribution:")
        print(results['predicted_quartile'].value_counts().sort_index())
    
    if 'confidence' in results.columns:
        print(f"\nAverage confidence: {results['confidence'].mean():.1f}%")
        print(f"Low confidence (<60%): {(results['confidence'] < 60).sum()}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()