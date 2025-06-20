"""
Fair and Equitable Evaluation Implementation
===========================================

This module implements bias-free evaluation of medical school applicants
using unstructured text from specific data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from typing import Dict, List, Optional, Tuple
import hashlib
from datetime import datetime

# For text anonymization
import spacy

# Load spacy model for named entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please install spacy model: python -m spacy download en_core_web_sm")
    nlp = None


class FairEvaluationPipeline:
    """
    Implements systematic, bias-free evaluation of medical applicants
    """
    
    def __init__(self, data_path: Path, year: int):
        self.data_path = data_path / f"{year} Applicants Reviewed by Trusted Reviewers"
        self.year = year
        
        # Define which files contain unstructured text for LLM
        self.unstructured_sources = {
            'personal_statement': {
                'file': '9. Personal Statement.xlsx',
                'text_column': 'personal_statement',
                'id_column': 'AMCAS_ID'
            },
            'secondary_essays': {
                'file': '10. Secondary Application.xlsx',
                'text_columns': [
                    '1 - Personal Attributes / Life Experiences',
                    '2 - Challenging Situation',
                    '3 - Reflect Experience',
                    '4 - Hope to Gain',
                    '6 - Experiences',
                    '7 - COVID Impact'
                ],
                'id_column': 'AMCAS ID'
            },
            'experiences': {
                'file': '6. Experiences.xlsx',
                'text_columns': ['Exp_Desc', 'Meaningful_Desc'],
                'id_column': 'AMCAS_ID'
            },
            'hardship': {
                'file': '1. Applicants.xlsx',
                'text_column': 'Hrdshp_Comments',
                'id_column': 'Amcas_ID',
                'conditional': True  # Only use if present
            }
        }
        
    def load_unstructured_content(self, applicant_id: str) -> Dict[str, any]:
        """
        Load all unstructured text for a single applicant
        """
        content = {}
        
        # Personal Statement
        ps_file = self.data_path / self.unstructured_sources['personal_statement']['file']
        if ps_file.exists():
            df = pd.read_excel(ps_file)
            id_col = self.unstructured_sources['personal_statement']['id_column']
            text_col = self.unstructured_sources['personal_statement']['text_column']
            
            applicant_row = df[df[id_col].astype(str) == str(applicant_id)]
            if not applicant_row.empty and text_col in df.columns:
                content['personal_statement'] = str(applicant_row.iloc[0][text_col])
        
        # Secondary Essays
        sec_file = self.data_path / self.unstructured_sources['secondary_essays']['file']
        if sec_file.exists():
            df = pd.read_excel(sec_file)
            id_col = self.unstructured_sources['secondary_essays']['id_column']
            
            applicant_row = df[df[id_col].astype(str) == str(applicant_id)]
            if not applicant_row.empty:
                content['secondary_essays'] = {}
                for col in self.unstructured_sources['secondary_essays']['text_columns']:
                    if col in df.columns:
                        essay_text = applicant_row.iloc[0][col]
                        if pd.notna(essay_text):
                            # Clean column name for key
                            key = col.split(' - ', 1)[1] if ' - ' in col else col
                            content['secondary_essays'][key] = str(essay_text)
        
        # Experience Descriptions
        exp_file = self.data_path / self.unstructured_sources['experiences']['file']
        if exp_file.exists():
            df = pd.read_excel(exp_file)
            id_col = self.unstructured_sources['experiences']['id_column']
            
            applicant_rows = df[df[id_col].astype(str) == str(applicant_id)]
            if not applicant_rows.empty:
                content['experiences'] = []
                for _, row in applicant_rows.iterrows():
                    exp_entry = {}
                    for col in self.unstructured_sources['experiences']['text_columns']:
                        if col in df.columns and pd.notna(row[col]):
                            exp_entry[col] = str(row[col])
                    if exp_entry:
                        content['experiences'].append(exp_entry)
        
        # Hardship Context (if present)
        hardship_file = self.data_path / self.unstructured_sources['hardship']['file']
        if hardship_file.exists():
            df = pd.read_excel(hardship_file)
            id_col = self.unstructured_sources['hardship']['id_column']
            text_col = self.unstructured_sources['hardship']['text_column']
            
            applicant_row = df[df[id_col].astype(str) == str(applicant_id)]
            if not applicant_row.empty and text_col in df.columns:
                hardship_text = applicant_row.iloc[0][text_col]
                if pd.notna(hardship_text) and str(hardship_text).strip():
                    content['hardship_context'] = str(hardship_text)
        
        return content
    
    def anonymize_text(self, text: str) -> str:
        """
        Remove identifying information from text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove specific years that might identify age
        current_year = datetime.now().year
        for year in range(current_year - 30, current_year + 5):
            text = text.replace(str(year), '[YEAR]')
        
        # Use spacy for named entity recognition if available
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                    if ent.label_ == 'PERSON':
                        text = text.replace(ent.text, '[NAME]')
                    elif ent.label_ == 'ORG':
                        # Keep generic medical terms
                        if not any(term in ent.text.lower() for term in 
                                 ['hospital', 'clinic', 'medical', 'health', 'university']):
                            text = text.replace(ent.text, '[ORGANIZATION]')
                    elif ent.label_ in ['GPE', 'LOC']:
                        text = text.replace(ent.text, '[LOCATION]')
        
        # Remove potential school names (unless generic)
        prestigious_schools = [
            'Harvard', 'Yale', 'Stanford', 'MIT', 'Princeton', 'Columbia',
            'Chicago', 'Duke', 'Penn', 'Cornell', 'Brown', 'Dartmouth'
        ]
        for school in prestigious_schools:
            text = re.sub(rf'\b{school}\b', '[UNIVERSITY]', text, flags=re.IGNORECASE)
        
        return text
    
    def prepare_for_llm_evaluation(self, content: Dict) -> str:
        """
        Prepare anonymized content for LLM evaluation
        """
        # Anonymize all text content
        anonymized = {}
        
        # Personal statement
        if 'personal_statement' in content:
            anonymized['personal_statement'] = self.anonymize_text(content['personal_statement'])
        
        # Secondary essays
        if 'secondary_essays' in content:
            anonymized['secondary_essays'] = {}
            for key, essay in content['secondary_essays'].items():
                anonymized['secondary_essays'][key] = self.anonymize_text(essay)
        
        # Experiences
        if 'experiences' in content:
            anonymized['experiences'] = []
            for exp in content['experiences']:
                anon_exp = {}
                for key, text in exp.items():
                    anon_exp[key] = self.anonymize_text(text)
                anonymized['experiences'].append(anon_exp)
        
        # Hardship context (be especially careful here)
        if 'hardship_context' in content:
            # Only include if it adds meaningful context
            hardship_anon = self.anonymize_text(content['hardship_context'])
            # Further remove any potential SES indicators
            hardship_anon = re.sub(r'\$[\d,]+', '[AMOUNT]', hardship_anon)
            hardship_anon = re.sub(r'\b\d+k\b', '[AMOUNT]', hardship_anon, flags=re.IGNORECASE)
            anonymized['hardship_context'] = hardship_anon
        
        return anonymized
    
    def create_fair_evaluation_prompt(self, anonymized_content: Dict) -> str:
        """
        Create evaluation prompt that prevents bias
        """
        prompt = """You are evaluating a medical school applicant based solely on the content and quality of their written materials. 

CRITICAL INSTRUCTIONS:
1. Evaluate ONLY based on demonstrated qualities, not demographics
2. Focus on growth, impact, and understanding shown
3. Value diverse paths to medicine equally
4. Consider achievements relative to challenges mentioned
5. Do NOT infer or consider applicant background beyond what is directly stated

CONTENT TO EVALUATE:

"""
        
        # Add personal statement
        if 'personal_statement' in anonymized_content:
            prompt += f"PERSONAL STATEMENT:\n{anonymized_content['personal_statement'][:3500]}\n\n"
        
        # Add secondary essays
        if 'secondary_essays' in anonymized_content:
            prompt += "SECONDARY ESSAYS:\n"
            for key, essay in anonymized_content['secondary_essays'].items():
                prompt += f"\n{key}:\n{essay[:800]}\n"
            prompt += "\n"
        
        # Add experiences
        if 'experiences' in anonymized_content:
            prompt += "KEY EXPERIENCES:\n"
            for i, exp in enumerate(anonymized_content['experiences'][:5]):  # Limit to 5
                prompt += f"\nExperience {i+1}:\n"
                if 'Exp_Desc' in exp:
                    prompt += f"Description: {exp['Exp_Desc'][:400]}\n"
                if 'Meaningful_Desc' in exp:
                    prompt += f"Why Meaningful: {exp['Meaningful_Desc'][:400]}\n"
            prompt += "\n"
        
        # Add hardship context if present
        if 'hardship_context' in anonymized_content:
            prompt += f"CONTEXT OF ACHIEVEMENTS:\n{anonymized_content['hardship_context'][:800]}\n\n"
        
        prompt += """
EVALUATION TASK:
Score each dimension 1-10 based ONLY on evidence in the text:
1. motivation_for_medicine: Authenticity and depth of calling
2. clinical_understanding: Insight from healthcare experiences  
3. service_commitment: Evidence of sustained altruism
4. resilience_growth: Learning from challenges
5. intellectual_curiosity: Academic engagement and growth
6. interpersonal_impact: Positive influence on others
7. leadership_initiative: Creating change or solutions
8. professional_maturity: Understanding of physician role

Also provide:
- overall_score (0-100)
- tier_recommendation (1-4)
- interview_decision (YES/NO)
- key_strengths (list 3-5)
- areas_for_growth (list 1-3)
- unique_contributions (what they bring to medicine)

Return as JSON. Focus on WHAT they learned and HOW they grew, not WHERE they did it or WHO they are."""
        
        return prompt
    
    def validate_evaluation_fairness(self, evaluation: Dict) -> Dict:
        """
        Check evaluation for potential bias indicators
        """
        bias_checks = {
            'demographic_inference': False,
            'institution_bias': False,
            'linguistic_bias': False,
            'experience_type_bias': False
        }
        
        # Check for demographic inference
        demographic_terms = ['male', 'female', 'ethnic', 'racial', 'minority', 'immigrant', 'foreign']
        eval_text = json.dumps(evaluation).lower()
        
        for term in demographic_terms:
            if term in eval_text:
                bias_checks['demographic_inference'] = True
                break
        
        # Check for institution bias
        prestige_terms = ['ivy', 'prestigious', 'top school', 'elite']
        for term in prestige_terms:
            if term in eval_text:
                bias_checks['institution_bias'] = True
                break
        
        # Add bias warnings to evaluation
        if any(bias_checks.values()):
            evaluation['bias_warnings'] = bias_checks
        
        return evaluation
    
    def generate_evaluation_report(self, applicant_id: str, evaluation: Dict) -> Dict:
        """
        Generate comprehensive evaluation report
        """
        report = {
            'applicant_id': applicant_id,
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_version': '2.0_fair',
            
            # Core scores
            'dimension_scores': {
                'motivation_for_medicine': evaluation.get('motivation_for_medicine', 5),
                'clinical_understanding': evaluation.get('clinical_understanding', 5),
                'service_commitment': evaluation.get('service_commitment', 5),
                'resilience_growth': evaluation.get('resilience_growth', 5),
                'intellectual_curiosity': evaluation.get('intellectual_curiosity', 5),
                'interpersonal_impact': evaluation.get('interpersonal_impact', 5),
                'leadership_initiative': evaluation.get('leadership_initiative', 5),
                'professional_maturity': evaluation.get('professional_maturity', 5)
            },
            
            # Overall assessment
            'overall_score': evaluation.get('overall_score', 50),
            'tier_recommendation': evaluation.get('tier_recommendation', 2),
            'interview_decision': evaluation.get('interview_decision', 'NO'),
            
            # Qualitative insights
            'key_strengths': evaluation.get('key_strengths', []),
            'areas_for_growth': evaluation.get('areas_for_growth', []),
            'unique_contributions': evaluation.get('unique_contributions', []),
            
            # Fairness metrics
            'evaluation_based_on': 'content_only',
            'bias_checks_passed': 'bias_warnings' not in evaluation,
            'anonymization_applied': True
        }
        
        # Add tier name
        tier_names = {
            1: 'Very Unlikely',
            2: 'Potential Review', 
            3: 'Probable Interview',
            4: 'Very Likely Interview'
        }
        report['tier_name'] = tier_names.get(report['tier_recommendation'], 'Unknown')
        
        return report


# Example usage
def evaluate_applicant_fairly(applicant_id: str, year: int = 2024) -> Dict:
    """
    Complete fair evaluation pipeline for a single applicant
    """
    # Initialize pipeline
    pipeline = FairEvaluationPipeline(
        data_path=Path("data"),
        year=year
    )
    
    # Load content
    content = pipeline.load_unstructured_content(applicant_id)
    
    # Anonymize
    anonymized = pipeline.prepare_for_llm_evaluation(content)
    
    # Create prompt
    prompt = pipeline.create_fair_evaluation_prompt(anonymized)
    
    # Here you would call your LLM (Azure OpenAI)
    # evaluation = call_azure_openai(prompt)
    
    # For demonstration, using placeholder
    evaluation = {
        'motivation_for_medicine': 8,
        'clinical_understanding': 7,
        'service_commitment': 8,
        'resilience_growth': 9,
        'intellectual_curiosity': 7,
        'interpersonal_impact': 8,
        'leadership_initiative': 7,
        'professional_maturity': 7,
        'overall_score': 76,
        'tier_recommendation': 3,
        'interview_decision': 'YES',
        'key_strengths': [
            'Demonstrated resilience overcoming significant challenges',
            'Created innovative community health program',
            'Deep reflection on clinical experiences'
        ],
        'areas_for_growth': [
            'Limited research exposure'
        ],
        'unique_contributions': [
            'Bridge between underserved community and medical profession'
        ]
    }
    
    # Validate for bias
    evaluation = pipeline.validate_evaluation_fairness(evaluation)
    
    # Generate report
    report = pipeline.generate_evaluation_report(applicant_id, evaluation)
    
    return report


if __name__ == "__main__":
    # Test the pipeline
    sample_report = evaluate_applicant_fairly("12345", 2024)
    print(json.dumps(sample_report, indent=2))