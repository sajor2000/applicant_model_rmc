"""
Step 2: Extract Unstructured Data from 2022-2023 Files
======================================================

This script extracts essays and experience descriptions from Excel files
for processing through Azure OpenAI.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnstructuredDataExtractor:
    """
    Extracts essays and unstructured text from medical admissions Excel files
    """
    
    def __init__(self, base_data_path: str = "data"):
        self.base_path = Path(base_data_path)
        
        # Define which files contain unstructured text
        self.unstructured_sources = {
            'personal_statement': {
                'file': '9. Personal Statement.xlsx',
                'id_column': 'AMCAS_ID',
                'text_column': 'personal_statement'
            },
            'secondary_application': {
                'file': '10. Secondary Application.xlsx',
                'id_column': 'AMCAS ID',  # Note: different from others
                'text_columns': [
                    '1 - Personal Attributes / Life Experiences',
                    '2 - Challenging Situation',
                    '3 - Reflect Experience',
                    '4 - Hope to Gain',
                    '6 - Experiences',
                    '7 - COVID Impact'
                ]
            },
            'experiences': {
                'file': '6. Experiences.xlsx',
                'id_column': 'AMCAS_ID',
                'text_columns': ['Exp_Desc', 'Meaningful_Desc']
            }
        }
    
    def extract_year_data(self, year: int) -> Dict[str, Dict]:
        """
        Extract all unstructured data for a given year
        
        Returns:
            Dictionary mapping AMCAS_ID to their unstructured content
        """
        logger.info(f"Extracting unstructured data for year {year}")
        
        year_path = self.base_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        if not year_path.exists():
            raise FileNotFoundError(f"Data directory not found: {year_path}")
        
        # Initialize storage
        applicant_data = {}
        
        # Extract personal statements
        ps_data = self._extract_personal_statements(year_path)
        for amcas_id, ps_text in ps_data.items():
            applicant_data[amcas_id] = {'personal_statement': ps_text}
        
        # Extract secondary essays
        secondary_data = self._extract_secondary_essays(year_path)
        for amcas_id, essays in secondary_data.items():
            if amcas_id in applicant_data:
                applicant_data[amcas_id]['secondary_essays'] = essays
            else:
                applicant_data[amcas_id] = {'secondary_essays': essays}
        
        # Extract experience descriptions
        exp_data = self._extract_experiences(year_path)
        for amcas_id, experiences in exp_data.items():
            if amcas_id in applicant_data:
                applicant_data[amcas_id]['experiences'] = experiences
            else:
                applicant_data[amcas_id] = {'experiences': experiences}
        
        logger.info(f"Extracted data for {len(applicant_data)} applicants from {year}")
        return applicant_data
    
    def _extract_personal_statements(self, year_path: Path) -> Dict[str, str]:
        """Extract personal statements"""
        ps_file = year_path / self.unstructured_sources['personal_statement']['file']
        
        if not ps_file.exists():
            logger.warning(f"Personal statement file not found: {ps_file}")
            return {}
        
        try:
            df = pd.read_excel(ps_file)
            id_col = self.unstructured_sources['personal_statement']['id_column']
            text_col = self.unstructured_sources['personal_statement']['text_column']
            
            # Check if columns exist
            if id_col not in df.columns or text_col not in df.columns:
                logger.error(f"Required columns not found. Available: {df.columns.tolist()}")
                return {}
            
            # Extract non-null personal statements
            ps_data = {}
            for _, row in df.iterrows():
                amcas_id = str(row[id_col])
                ps_text = row[text_col]
                
                if pd.notna(ps_text) and str(ps_text).strip():
                    ps_data[amcas_id] = str(ps_text).strip()
            
            logger.info(f"Extracted {len(ps_data)} personal statements")
            return ps_data
            
        except Exception as e:
            logger.error(f"Error extracting personal statements: {e}")
            return {}
    
    def _extract_secondary_essays(self, year_path: Path) -> Dict[str, Dict[str, str]]:
        """Extract secondary application essays"""
        sec_file = year_path / self.unstructured_sources['secondary_application']['file']
        
        if not sec_file.exists():
            logger.warning(f"Secondary application file not found: {sec_file}")
            return {}
        
        try:
            df = pd.read_excel(sec_file)
            id_col = self.unstructured_sources['secondary_application']['id_column']
            text_columns = self.unstructured_sources['secondary_application']['text_columns']
            
            # Check if ID column exists
            if id_col not in df.columns:
                # Try alternative column names
                alt_id_cols = ['AMCAS_ID', 'Amcas_ID', 'AMCAS ID']
                for alt_col in alt_id_cols:
                    if alt_col in df.columns:
                        id_col = alt_col
                        break
                else:
                    logger.error(f"No ID column found. Available: {df.columns.tolist()}")
                    return {}
            
            # Extract essays for each applicant
            essays_data = {}
            for _, row in df.iterrows():
                amcas_id = str(row[id_col])
                applicant_essays = {}
                
                # Extract each essay type
                for col in text_columns:
                    if col in df.columns:
                        essay_text = row[col]
                        if pd.notna(essay_text) and str(essay_text).strip():
                            applicant_essays[col] = str(essay_text).strip()
                
                if applicant_essays:  # Only add if at least one essay exists
                    essays_data[amcas_id] = applicant_essays
            
            logger.info(f"Extracted secondary essays for {len(essays_data)} applicants")
            return essays_data
            
        except Exception as e:
            logger.error(f"Error extracting secondary essays: {e}")
            return {}
    
    def _extract_experiences(self, year_path: Path) -> Dict[str, str]:
        """Extract and format experience descriptions"""
        exp_file = year_path / self.unstructured_sources['experiences']['file']
        
        if not exp_file.exists():
            logger.warning(f"Experiences file not found: {exp_file}")
            return {}
        
        try:
            df = pd.read_excel(exp_file)
            id_col = self.unstructured_sources['experiences']['id_column']
            text_columns = self.unstructured_sources['experiences']['text_columns']
            
            # Check columns
            if id_col not in df.columns:
                logger.error(f"ID column {id_col} not found")
                return {}
            
            # Group by applicant and format experiences
            exp_data = {}
            
            for amcas_id, group in df.groupby(id_col):
                amcas_id = str(amcas_id)
                experiences_text = ""
                
                for idx, (_, row) in enumerate(group.iterrows(), 1):
                    # Add experience description
                    if 'Exp_Desc' in df.columns and pd.notna(row['Exp_Desc']):
                        experiences_text += f"\nExperience {idx}:\n"
                        experiences_text += f"Description: {str(row['Exp_Desc']).strip()}\n"
                    
                    # Add meaningful description if available
                    if 'Meaningful_Desc' in df.columns and pd.notna(row['Meaningful_Desc']):
                        experiences_text += f"Why Meaningful: {str(row['Meaningful_Desc']).strip()}\n"
                
                if experiences_text.strip():
                    exp_data[amcas_id] = experiences_text.strip()
            
            logger.info(f"Extracted experiences for {len(exp_data)} applicants")
            return exp_data
            
        except Exception as e:
            logger.error(f"Error extracting experiences: {e}")
            return {}
    
    def extract_multiple_years(self, years: List[int]) -> Dict[str, Dict]:
        """
        Extract data for multiple years and combine
        
        Args:
            years: List of years to extract (e.g., [2022, 2023])
            
        Returns:
            Combined dictionary with year prefixed to AMCAS_ID
        """
        all_data = {}
        
        for year in years:
            try:
                year_data = self.extract_year_data(year)
                
                # Prefix AMCAS_ID with year to handle duplicates
                for amcas_id, content in year_data.items():
                    key = f"{year}_{amcas_id}"
                    all_data[key] = content
                    all_data[key]['year'] = year
                    
            except Exception as e:
                logger.error(f"Failed to extract {year} data: {e}")
                continue
        
        return all_data
    
    def get_summary_statistics(self, data: Dict[str, Dict]) -> Dict:
        """Get summary statistics about extracted data"""
        stats = {
            'total_applicants': len(data),
            'has_personal_statement': 0,
            'has_secondary_essays': 0,
            'has_experiences': 0,
            'complete_data': 0,
            'avg_ps_length': [],
            'avg_secondary_count': []
        }
        
        for amcas_id, content in data.items():
            if 'personal_statement' in content:
                stats['has_personal_statement'] += 1
                stats['avg_ps_length'].append(len(content['personal_statement']))
            
            if 'secondary_essays' in content:
                stats['has_secondary_essays'] += 1
                stats['avg_secondary_count'].append(len(content['secondary_essays']))
            
            if 'experiences' in content:
                stats['has_experiences'] += 1
            
            # Check if has all three
            if all(key in content for key in ['personal_statement', 'secondary_essays', 'experiences']):
                stats['complete_data'] += 1
        
        # Calculate averages
        if stats['avg_ps_length']:
            stats['avg_ps_length'] = np.mean(stats['avg_ps_length'])
        else:
            stats['avg_ps_length'] = 0
            
        if stats['avg_secondary_count']:
            stats['avg_secondary_count'] = np.mean(stats['avg_secondary_count'])
        else:
            stats['avg_secondary_count'] = 0
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = UnstructuredDataExtractor("data")
    
    # Extract 2022 and 2023 data
    print("Extracting unstructured data for 2022-2023...")
    all_data = extractor.extract_multiple_years([2022, 2023])
    
    # Get statistics
    stats = extractor.get_summary_statistics(all_data)
    
    print("\nExtraction Summary:")
    print(f"Total applicants: {stats['total_applicants']}")
    print(f"With personal statements: {stats['has_personal_statement']}")
    print(f"With secondary essays: {stats['has_secondary_essays']}")
    print(f"With experiences: {stats['has_experiences']}")
    print(f"Complete data: {stats['complete_data']}")
    print(f"Average PS length: {stats['avg_ps_length']:.0f} characters")
    print(f"Average secondary essays: {stats['avg_secondary_count']:.1f}")
    
    # Show sample data structure
    if all_data:
        sample_id = list(all_data.keys())[0]
        print(f"\nSample data structure for {sample_id}:")
        sample = all_data[sample_id]
        
        if 'personal_statement' in sample:
            print(f"- Personal Statement: {len(sample['personal_statement'])} chars")
        if 'secondary_essays' in sample:
            print(f"- Secondary Essays: {list(sample['secondary_essays'].keys())}")
        if 'experiences' in sample:
            print(f"- Experiences: {len(sample['experiences'])} chars")