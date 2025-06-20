"""
Explore where clinical experience hours are stored in the admissions data
"""

import pandas as pd
import os

def explore_applicants_file():
    """Check applicants file for clinical hours columns"""
    print("=== EXPLORING APPLICANTS FILE ===")
    
    # Load the applicants file
    file_path = "data_standardized/2022_standardized/1. Applicants.xlsx"
    df = pd.read_excel(file_path)
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Look for columns that might contain clinical hours
    clinical_keywords = ['clinical', 'patient', 'healthcare', 'medical', 'care', 'exp', 'hour', 
                        'volunteer', 'shadow', 'employ', 'work', 'service', 'hospital', 'clinic']
    
    print("\n=== COLUMNS POTENTIALLY CONTAINING CLINICAL HOURS ===")
    relevant_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in clinical_keywords):
            relevant_columns.append(col)
            print(f"\n{col}:")
            # Show some sample values
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                print(f"  - Data type: {df[col].dtype}")
                print(f"  - Non-null count: {len(non_null_values)}")
                print(f"  - Sample values: {non_null_values.head(3).tolist()}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"  - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")
    
    return relevant_columns


def explore_experiences_file():
    """Check experiences file structure"""
    print("\n\n=== EXPLORING EXPERIENCES FILE ===")
    
    # Load the experiences file
    file_path = "data_standardized/2022_standardized/6. Experiences.xlsx"
    df = pd.read_excel(file_path)
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"\nColumn names: {df.columns.tolist()}")
    
    # Show some sample data
    print("\n=== SAMPLE DATA FROM EXPERIENCES FILE ===")
    print(df.head())
    
    # Check for experience types
    if 'Exp_Type' in df.columns:
        print("\n=== EXPERIENCE TYPES ===")
        print(df['Exp_Type'].value_counts())
    
    # Check for hours columns
    hours_columns = [col for col in df.columns if 'hour' in col.lower() or 'hrs' in col.lower()]
    if hours_columns:
        print("\n=== HOURS COLUMNS ===")
        for col in hours_columns:
            print(f"\n{col}:")
            print(f"  - Non-null count: {df[col].notna().sum()}")
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"  - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")
    
    # Check how many experiences per applicant
    if 'AMCAS_ID' in df.columns:
        print("\n=== EXPERIENCES PER APPLICANT ===")
        exp_per_applicant = df.groupby('AMCAS_ID').size()
        print(f"Average experiences per applicant: {exp_per_applicant.mean():.2f}")
        print(f"Min: {exp_per_applicant.min()}, Max: {exp_per_applicant.max()}")
    
    return df


def analyze_clinical_hours_calculation():
    """Analyze how to calculate clinical hours from the data"""
    print("\n\n=== ANALYSIS: HOW TO GET CLINICAL HOURS ===")
    
    # Load applicants file
    applicants_df = pd.read_excel("data_standardized/2022_standardized/1. Applicants.xlsx")
    
    # Identify medical/clinical experience columns in applicants file
    medical_cols = []
    for col in applicants_df.columns:
        col_lower = col.lower()
        if ('exp_hour' in col_lower or 'healthcare' in col_lower) and 'hour' in col_lower:
            medical_cols.append(col)
    
    print("\n1. DIRECT CLINICAL HOURS COLUMNS IN APPLICANTS FILE:")
    for col in medical_cols:
        if any(term in col.lower() for term in ['med', 'healthcare', 'shadow', 'clinical']):
            print(f"   - {col}")
            non_null = applicants_df[col].notna().sum()
            if non_null > 0:
                print(f"     Non-null: {non_null}, Mean: {applicants_df[col].mean():.2f}")
    
    # Check if we need to aggregate from experiences file
    try:
        exp_df = pd.read_excel("data_standardized/2022_standardized/6. Experiences.xlsx")
        
        print("\n2. EXPERIENCE TYPES THAT MIGHT BE CLINICAL:")
        if 'Exp_Type' in exp_df.columns:
            clinical_types = []
            for exp_type in exp_df['Exp_Type'].unique():
                if exp_type and any(term in str(exp_type).lower() for term in 
                                  ['medical', 'clinical', 'patient', 'healthcare', 'hospital', 
                                   'physician', 'shadow', 'scribe']):
                    clinical_types.append(exp_type)
                    count = (exp_df['Exp_Type'] == exp_type).sum()
                    print(f"   - {exp_type}: {count} entries")
            
            # Calculate total clinical hours by aggregating
            if clinical_types and 'Total_Hours' in exp_df.columns:
                clinical_exp = exp_df[exp_df['Exp_Type'].isin(clinical_types)]
                clinical_hours_by_applicant = clinical_exp.groupby('AMCAS_ID')['Total_Hours'].sum()
                print(f"\n   Applicants with clinical experiences: {len(clinical_hours_by_applicant)}")
                print(f"   Average clinical hours (from experiences): {clinical_hours_by_applicant.mean():.2f}")
    except Exception as e:
        print(f"\n   Could not analyze experiences file: {e}")
    
    print("\n3. RECOMMENDATION:")
    print("   Clinical hours appear to be stored in TWO places:")
    print("   a) Pre-calculated in Applicants.xlsx:")
    print("      - Exp_Hour_Volunteer_Med (Medical/Clinical Volunteering)")
    print("      - Exp_Hour_Shadowing (Physician Shadowing)")
    print("      - Exp_Hour_Employ_Med (Medical/Clinical Employment)")
    print("      - HealthCare_Total_Hours (Total Healthcare Experience)")
    print("   b) Can be calculated from 6. Experiences.xlsx by:")
    print("      - Filtering for clinical experience types")
    print("      - Summing Total_Hours for each applicant")
    print("\n   The applicants file appears to have pre-aggregated values,")
    print("   which is more convenient for modeling.")


if __name__ == "__main__":
    # Explore both files
    applicants_columns = explore_applicants_file()
    experiences_df = explore_experiences_file()
    
    # Provide analysis
    analyze_clinical_hours_calculation()