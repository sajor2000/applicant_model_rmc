"""
Filter and Keep Consistent Features
===================================

This script filters structured data to keep only features that:
1. Are consistent across all years (even if semantically different)
2. Have less than 75% missing data
3. Can be cleaned/standardized across years
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json


class ConsistentFeatureFilter:
    """Filters features based on consistency and completeness criteria."""
    
    def __init__(self, base_path: str = "data_standardized", 
                 missing_threshold: float = 0.75):
        self.base_path = Path(base_path)
        self.missing_threshold = missing_threshold
        self.years = [2022, 2023, 2024]
        
    def analyze_feature_consistency(self) -> Tuple[Dict, Dict]:
        """Analyze features across all years for consistency and missing rates."""
        feature_stats = {}
        year_data = {}
        
        # Load all years' data
        for year in self.years:
            file_path = self.base_path / f"{year}_standardized" / "1. Applicants.xlsx"
            if file_path.exists():
                df = pd.read_excel(file_path)
                year_data[year] = df
                
                # Calculate missing rates for each feature
                for col in df.columns:
                    if col not in feature_stats:
                        feature_stats[col] = {
                            'years_present': [],
                            'missing_rates': {},
                            'dtypes': {},
                            'unique_values': {}
                        }
                    
                    feature_stats[col]['years_present'].append(year)
                    feature_stats[col]['missing_rates'][year] = df[col].isnull().sum() / len(df)
                    feature_stats[col]['dtypes'][year] = str(df[col].dtype)
                    
                    # Sample unique values for categorical features
                    if df[col].dtype == 'object':
                        unique_vals = df[col].dropna().unique()[:10]
                        feature_stats[col]['unique_values'][year] = list(unique_vals)
        
        return feature_stats, year_data
    
    def identify_consistent_features(self, feature_stats: Dict) -> List[str]:
        """Identify features that meet consistency criteria."""
        consistent_features = []
        
        for feature, stats in feature_stats.items():
            # Check if present in all years
            if len(stats['years_present']) != len(self.years):
                continue
            
            # Check if missing rate is below threshold for all years
            max_missing = max(stats['missing_rates'].values())
            if max_missing > self.missing_threshold:
                continue
            
            # Feature passes criteria
            consistent_features.append(feature)
        
        return consistent_features
    
    def create_semantic_mappings(self, feature_stats: Dict) -> Dict:
        """Create mappings for semantically similar features across years."""
        semantic_mappings = {
            # Yes/No to binary
            'yes_no_to_binary': {
                'Yes': 1, 'Y': 1, 'yes': 1,
                'No': 0, 'N': 0, 'no': 0,
                'Unknown': np.nan, '': np.nan, None: np.nan
            },
            
            # Gender standardization
            'gender_mapping': {
                'M': 'Male', 'Male': 'Male', 'male': 'Male',
                'F': 'Female', 'Female': 'Female', 'female': 'Female',
                'O': 'Other', 'Other': 'Other', 'other': 'Other',
                'U': 'Unknown', 'Unknown': 'Unknown', '': 'Unknown'
            },
            
            # Citizenship standardization
            'citizenship_mapping': {
                'US Citizen': 'US_Citizen',
                'US citizen': 'US_Citizen',
                'Permanent Resident': 'Permanent_Resident',
                'Permanent resident': 'Permanent_Resident',
                'International': 'International',
                'Other': 'Other'
            }
        }
        
        return semantic_mappings
    
    def clean_and_standardize_data(self, df: pd.DataFrame, 
                                   feature_list: List[str],
                                   mappings: Dict) -> pd.DataFrame:
        """Clean and standardize data for consistent features."""
        df_clean = df[feature_list].copy()
        
        # Apply mappings to binary features
        binary_features = [
            'first_generation_ind', 'disadvantaged_ind', 'pell_grant',
            'fee_assistance_program', 'felony_ind', 'misdemeanor_ind',
            'inst_action_ind', 'investigation_ind', 'healthcare_ind',
            'comm_service_ind', 'military_service', 'ru_ind'
        ]
        
        for col in binary_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(
                    lambda x: mappings['yes_no_to_binary'].get(x, x)
                )
        
        # Standardize gender
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].map(
                lambda x: mappings['gender_mapping'].get(x, x)
            )
        
        # Standardize citizenship
        if 'citizenship' in df_clean.columns:
            df_clean['citizenship'] = df_clean['citizenship'].map(
                lambda x: mappings['citizenship_mapping'].get(x, x)
            )
        
        # Convert numeric columns stored as strings
        numeric_cols = [
            'age', 'exp_hour_total', 'exp_hour_research',
            'exp_hour_volunteer_med', 'exp_hour_volunteer_non_med',
            'exp_hour_employ_med', 'exp_hour_shadowing',
            'healthcare_total_hours', 'comm_service_total_hours',
            'ses_value', 'eo_level', 'student_loan_percentage',
            'academic_scholarship_percentage', 'number_in_household',
            'num_dependents', 'service_rating_numerical',
            'application_review_score'
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def save_filtered_data(self, consistent_features: List[str], 
                          year_data: Dict, mappings: Dict):
        """Save filtered and cleaned data for each year."""
        output_base = Path("data_filtered")
        output_base.mkdir(exist_ok=True)
        
        for year, df in year_data.items():
            # Get features that exist in this year
            available_features = [f for f in consistent_features if f in df.columns]
            
            # Clean and standardize
            df_clean = self.clean_and_standardize_data(df, available_features, mappings)
            
            # Save
            output_path = output_base / f"{year}_filtered_applicants.xlsx"
            df_clean.to_excel(output_path, index=False)
            print(f"✓ Saved {year} filtered data: {len(available_features)} features, {len(df_clean)} rows")
    
    def create_feature_report(self, feature_stats: Dict, 
                             consistent_features: List[str]) -> str:
        """Create a report of filtered features."""
        report = []
        report.append("="*80)
        report.append("CONSISTENT FEATURES REPORT")
        report.append("="*80)
        
        # Summary
        total_features = len(feature_stats)
        kept_features = len(consistent_features)
        report.append(f"\nTotal features analyzed: {total_features}")
        report.append(f"Features kept (consistent & <75% missing): {kept_features}")
        report.append(f"Features removed: {total_features - kept_features}")
        
        # Kept features by category
        report.append("\n## KEPT FEATURES BY CATEGORY:")
        
        categories = {
            'Demographics': ['age', 'gender', 'citizenship'],
            'Socioeconomic': ['first_generation_ind', 'disadvantaged_ind', 
                             'ses_value', 'eo_level', 'pell_grant', 
                             'fee_assistance_program'],
            'Experience Hours': ['exp_hour_total', 'exp_hour_research',
                               'exp_hour_volunteer_med', 'exp_hour_volunteer_non_med',
                               'exp_hour_employ_med', 'exp_hour_shadowing',
                               'healthcare_total_hours', 'comm_service_total_hours'],
            'Financial': ['student_loan_percentage', 'academic_scholarship_percentage'],
            'Application Info': ['appl_year', 'service_rating_numerical',
                               'service_rating_categorical', 'application_review_score'],
            'Flags': ['felony_ind', 'misdemeanor_ind', 'inst_action_ind',
                     'investigation_ind'],
            'Other': ['healthcare_ind', 'comm_service_ind', 'military_service',
                     'ru_ind', 'number_in_household', 'num_dependents']
        }
        
        for category, features in categories.items():
            kept_in_category = [f for f in features if f in consistent_features]
            if kept_in_category:
                report.append(f"\n### {category} ({len(kept_in_category)} features):")
                for feature in kept_in_category:
                    max_missing = max(feature_stats[feature]['missing_rates'].values())
                    report.append(f"  - {feature} (max {max_missing*100:.1f}% missing)")
        
        # Removed features
        report.append("\n## REMOVED FEATURES (>75% missing or inconsistent):")
        removed_features = []
        for feature, stats in feature_stats.items():
            if feature not in consistent_features:
                reason = []
                if len(stats['years_present']) != len(self.years):
                    reason.append(f"only in {stats['years_present']}")
                else:
                    max_missing = max(stats['missing_rates'].values())
                    if max_missing > self.missing_threshold:
                        reason.append(f"{max_missing*100:.1f}% missing")
                removed_features.append((feature, ', '.join(reason)))
        
        # Show first 20 removed features
        for feature, reason in removed_features[:20]:
            report.append(f"  - {feature}: {reason}")
        if len(removed_features) > 20:
            report.append(f"  ... and {len(removed_features)-20} more")
        
        # Missing data summary for kept features
        report.append("\n## MISSING DATA SUMMARY FOR KEPT FEATURES:")
        missing_summary = []
        for feature in consistent_features:
            max_missing = max(feature_stats[feature]['missing_rates'].values()) * 100
            avg_missing = np.mean(list(feature_stats[feature]['missing_rates'].values())) * 100
            if avg_missing > 10:  # Only show if >10% missing on average
                missing_summary.append((feature, avg_missing, max_missing))
        
        missing_summary.sort(key=lambda x: x[1], reverse=True)
        for feature, avg_miss, max_miss in missing_summary[:15]:
            report.append(f"  - {feature}: avg {avg_miss:.1f}%, max {max_miss:.1f}%")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    print("Filtering Consistent Features Across All Years")
    print("="*80)
    
    # Initialize filter
    filter = ConsistentFeatureFilter(missing_threshold=0.75)
    
    # Analyze features
    print("\n1. Analyzing feature consistency...")
    feature_stats, year_data = filter.analyze_feature_consistency()
    
    # Identify consistent features
    print("\n2. Identifying features with <75% missing data...")
    consistent_features = filter.identify_consistent_features(feature_stats)
    
    # Create semantic mappings
    print("\n3. Creating semantic mappings for standardization...")
    mappings = filter.create_semantic_mappings(feature_stats)
    
    # Save filtered data
    print("\n4. Saving filtered and cleaned data...")
    filter.save_filtered_data(consistent_features, year_data, mappings)
    
    # Create report
    report = filter.create_feature_report(feature_stats, consistent_features)
    print(report)
    
    # Save report
    with open("consistent_features_report.txt", "w") as f:
        f.write(report)
    print("\n✓ Report saved to consistent_features_report.txt")
    
    # Save feature list
    with open("consistent_features_list.json", "w") as f:
        json.dump({
            'consistent_features': consistent_features,
            'total_features': len(consistent_features),
            'semantic_mappings': mappings
        }, f, indent=2)
    print("✓ Feature list saved to consistent_features_list.json")
    
    # Summary
    print("\n" + "!"*80)
    print("FILTERING COMPLETE:")
    print(f"- Kept {len(consistent_features)} consistent features")
    print(f"- All features have <75% missing data across all years")
    print(f"- Data cleaned and standardized (Yes/No → 1/0, etc.)")
    print(f"- Ready for model training with high-quality features only")
    print("!"*80)


if __name__ == "__main__":
    main()