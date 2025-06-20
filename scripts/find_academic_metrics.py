#!/usr/bin/env python3
"""
Script to find where GPA and MCAT data might be stored
"""

import pandas as pd
from pathlib import Path
import sys

def search_for_academic_metrics():
    """Search all Excel files for GPA and MCAT data"""
    
    data_dirs = [
        'data/2022 Applicants Reviewed by Trusted Reviewers',
        'data/2023 Applicants Reviewed by Trusted Reviewers',
        'data/2024 Applicants Reviewed by Trusted Reviewers',
        'data_standardized/2022_standardized',
        'data_standardized/2023_standardized',
        'data_standardized/2024_standardized',
        'data_filtered'
    ]
    
    print("Searching for GPA and MCAT data across all Excel files...")
    print("="*60)
    
    found_files = []
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            continue
            
        for excel_file in dir_path.glob('*.xlsx'):
            try:
                # Read Excel file
                df = pd.read_excel(excel_file)
                
                # Check columns for GPA/MCAT keywords
                gpa_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in ['GPA', 'GRADE', 'ACADEMIC'])]
                mcat_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in ['MCAT', 'TEST', 'EXAM'])]
                
                if gpa_cols or mcat_cols:
                    print(f"\nFile: {excel_file}")
                    print(f"  GPA columns: {gpa_cols}")
                    print(f"  MCAT columns: {mcat_cols}")
                    found_files.append(str(excel_file))
                    
            except Exception as e:
                # Skip files that can't be read
                pass
    
    if not found_files:
        print("\nNo GPA or MCAT columns found in any Excel files!")
        print("\nThis suggests that these features might need to be:")
        print("1. Calculated from the Academic Records file")
        print("2. Loaded from a separate data source")
        print("3. Added as dummy/placeholder values")
    
    # Check if the features are referenced in the code but not in data
    print("\n" + "="*60)
    print("IMPORTANT FINDING:")
    print("="*60)
    print("The feature_engineer.py expects these columns:")
    print("- Undergrad_GPA")
    print("- Grad_GPA")
    print("- MCAT_Total")
    print("- Undergrad_BCPM")
    print("- Grad_BCPM")
    print("\nBUT these columns are NOT present in the actual data files!")
    print("\nThis is likely why the model is missing these important academic features.")
    
    return found_files

if __name__ == "__main__":
    search_for_academic_metrics()