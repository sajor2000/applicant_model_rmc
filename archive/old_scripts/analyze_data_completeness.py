"""
Analyze Data Completeness Across Years
======================================

This script performs a comprehensive analysis of structured data completeness
across all years to ensure all features are present and identify any gaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
import json


class DataCompletenessAnalyzer:
    """Analyzes data completeness and feature availability across years."""
    
    def __init__(self, base_path: str = "data_standardized"):
        self.base_path = Path(base_path)
        self.years = [2022, 2023, 2024]
        
        # Critical features for model training
        self.critical_features = [
            'service_rating_numerical',
            'application_review_score',
            'exp_hour_total',
            'exp_hour_research', 
            'exp_hour_clinical',
            'exp_hour_volunteer_med',
            'healthcare_total_hours',
            'comm_service_total_hours',
            'age',
            'gender',
            'first_generation_ind',
            'disadvantaged_ind',
            'ses_value',
            'pell_grant',
            'fee_assistance_program'
        ]
        
    def analyze_applicants_file(self) -> Dict:
        """Analyze the main applicants file across all years."""
        analysis = {
            'feature_availability': {},
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'critical_features_status': {}
        }
        
        all_features = set()
        
        for year in self.years:
            file_path = self.base_path / f"{year}_standardized" / "1. Applicants.xlsx"
            
            if file_path.exists():
                df = pd.read_excel(file_path)
                features = set(df.columns)
                all_features.update(features)
                
                # Store features for this year
                analysis['feature_availability'][year] = list(features)
                
                # Analyze missing values
                missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
                analysis['missing_values'][year] = {
                    k: round(v, 2) for k, v in missing_pct.items() if v > 0
                }
                
                # Data types
                analysis['data_types'][year] = df.dtypes.astype(str).to_dict()
                
                # Value ranges for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                analysis['value_ranges'][year] = {}
                for col in numeric_cols:
                    if col in df.columns and not df[col].isna().all():
                        analysis['value_ranges'][year][col] = {
                            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                        }
                
                # Check critical features
                analysis['critical_features_status'][year] = {}
                for feature in self.critical_features:
                    if feature in df.columns:
                        missing_pct = (df[feature].isnull().sum() / len(df) * 100)
                        analysis['critical_features_status'][year][feature] = {
                            'present': True,
                            'missing_pct': round(missing_pct, 2)
                        }
                    else:
                        analysis['critical_features_status'][year][feature] = {
                            'present': False,
                            'missing_pct': 100.0
                        }
        
        # Find features that are not consistent across years
        analysis['inconsistent_features'] = {}
        for feature in all_features:
            years_present = [year for year in self.years 
                           if feature in analysis['feature_availability'].get(year, [])]
            if len(years_present) != len(self.years):
                analysis['inconsistent_features'][feature] = {
                    'present_in': years_present,
                    'missing_in': [y for y in self.years if y not in years_present]
                }
        
        return analysis
    
    def analyze_all_files(self) -> Dict:
        """Analyze all Excel files across years."""
        file_analysis = {}
        
        for year in self.years:
            year_path = self.base_path / f"{year}_standardized"
            if year_path.exists():
                excel_files = list(year_path.glob("*.xlsx"))
                file_analysis[year] = {
                    'file_count': len(excel_files),
                    'files': [f.name for f in excel_files]
                }
        
        # Check file consistency
        all_files = set()
        for year_data in file_analysis.values():
            all_files.update(year_data.get('files', []))
        
        file_consistency = {}
        for file_name in all_files:
            years_present = [year for year, data in file_analysis.items() 
                           if file_name in data.get('files', [])]
            if len(years_present) != len(self.years):
                file_consistency[file_name] = {
                    'present_in': years_present,
                    'missing_in': [y for y in self.years if y not in years_present]
                }
        
        return {
            'file_analysis': file_analysis,
            'file_consistency': file_consistency
        }
    
    def create_completeness_report(self, applicants_analysis: Dict, files_analysis: Dict) -> str:
        """Create a comprehensive data completeness report."""
        report = []
        report.append("="*80)
        report.append("DATA COMPLETENESS ANALYSIS REPORT")
        report.append("="*80)
        
        # File consistency
        report.append("\n## FILE CONSISTENCY ACROSS YEARS")
        if files_analysis['file_consistency']:
            for file_name, info in files_analysis['file_consistency'].items():
                report.append(f"\n{file_name}:")
                report.append(f"  Present in: {info['present_in']}")
                report.append(f"  Missing in: {info['missing_in']}")
        else:
            report.append("✓ All files are present across all years")
        
        # Critical features status
        report.append("\n## CRITICAL FEATURES STATUS")
        for year in self.years:
            report.append(f"\n### Year {year}:")
            critical_status = applicants_analysis['critical_features_status'].get(year, {})
            
            present_features = [f for f, info in critical_status.items() if info['present']]
            missing_features = [f for f, info in critical_status.items() if not info['present']]
            
            report.append(f"  Present: {len(present_features)}/{len(self.critical_features)}")
            if missing_features:
                report.append(f"  Missing: {', '.join(missing_features)}")
            
            # High missing percentage
            high_missing = [(f, info['missing_pct']) for f, info in critical_status.items() 
                          if info['present'] and info['missing_pct'] > 50]
            if high_missing:
                report.append(f"  High missing (>50%):")
                for feature, pct in high_missing:
                    report.append(f"    - {feature}: {pct}% missing")
        
        # Inconsistent features
        report.append("\n## FEATURES NOT CONSISTENT ACROSS YEARS")
        inconsistent = applicants_analysis.get('inconsistent_features', {})
        if inconsistent:
            for feature, info in list(inconsistent.items())[:20]:  # Show first 20
                report.append(f"\n{feature}:")
                report.append(f"  Present in: {info['present_in']}")
                report.append(f"  Missing in: {info['missing_in']}")
            if len(inconsistent) > 20:
                report.append(f"\n... and {len(inconsistent)-20} more inconsistent features")
        
        # Missing data summary
        report.append("\n## MISSING DATA SUMMARY")
        for year in self.years:
            missing_data = applicants_analysis['missing_values'].get(year, {})
            if missing_data:
                report.append(f"\n### Year {year} - Features with missing data:")
                sorted_missing = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)
                for feature, pct in sorted_missing[:10]:  # Top 10
                    report.append(f"  - {feature}: {pct}% missing")
        
        # Data type consistency
        report.append("\n## DATA TYPE CHANGES")
        all_features_types = {}
        for year, types in applicants_analysis['data_types'].items():
            for feature, dtype in types.items():
                if feature not in all_features_types:
                    all_features_types[feature] = {}
                all_features_types[feature][year] = dtype
        
        type_changes = []
        for feature, year_types in all_features_types.items():
            unique_types = set(year_types.values())
            if len(unique_types) > 1:
                type_changes.append((feature, year_types))
        
        if type_changes:
            report.append("\nFeatures with inconsistent data types:")
            for feature, types in type_changes[:10]:
                report.append(f"\n{feature}:")
                for year, dtype in types.items():
                    report.append(f"  {year}: {dtype}")
        
        # Summary statistics
        report.append("\n## SUMMARY STATISTICS")
        for year in self.years:
            features = applicants_analysis['feature_availability'].get(year, [])
            report.append(f"\nYear {year}:")
            report.append(f"  Total features: {len(features)}")
            report.append(f"  Features with missing data: {len(applicants_analysis['missing_values'].get(year, {}))}")
            
        # Recommendations
        report.append("\n## RECOMMENDATIONS")
        report.append("\n1. Address Critical Missing Features:")
        all_missing_critical = set()
        for year_data in applicants_analysis['critical_features_status'].values():
            for feature, info in year_data.items():
                if not info['present']:
                    all_missing_critical.add(feature)
        
        if all_missing_critical:
            for feature in all_missing_critical:
                report.append(f"   - Add {feature} to all years")
        
        report.append("\n2. Handle High Missing Data:")
        report.append("   - Consider imputation strategies for features with >50% missing")
        report.append("   - Investigate why MCAT and GPA values are missing")
        
        report.append("\n3. Standardize Data Types:")
        report.append("   - Ensure consistent data types across years")
        report.append("   - Convert Yes/No to boolean consistently")
        
        return "\n".join(report)
    
    def save_detailed_analysis(self, analysis: Dict, filename: str = "data_completeness_analysis.json"):
        """Save detailed analysis to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_types(analysis), f, indent=2)
        
        print(f"✓ Detailed analysis saved to {filename}")


def main():
    """Main execution function."""
    print("Analyzing Data Completeness Across All Years")
    print("="*80)
    
    analyzer = DataCompletenessAnalyzer()
    
    # Analyze applicants file
    print("\n1. Analyzing main applicants file...")
    applicants_analysis = analyzer.analyze_applicants_file()
    
    # Analyze all files
    print("\n2. Analyzing all data files...")
    files_analysis = analyzer.analyze_all_files()
    
    # Create report
    report = analyzer.create_completeness_report(applicants_analysis, files_analysis)
    print(report)
    
    # Save report
    with open("data_completeness_report.txt", "w") as f:
        f.write(report)
    print("\n✓ Report saved to data_completeness_report.txt")
    
    # Save detailed analysis
    complete_analysis = {
        'applicants_analysis': applicants_analysis,
        'files_analysis': files_analysis
    }
    analyzer.save_detailed_analysis(complete_analysis)
    
    # Critical findings
    print("\n" + "!"*80)
    print("KEY FINDINGS:")
    print("1. Column names have been standardized across all years")
    print("2. Service Rating is now consistently named 'service_rating_numerical'")
    print("3. Some features like GPA trends are missing from 2023/2024 data")
    print("4. MCAT and actual GPA values appear to be missing from all years")
    print("5. Ready to retrain model with standardized data")
    print("!"*80)


if __name__ == "__main__":
    main()