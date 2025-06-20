"""Essay analysis using Azure OpenAI GPT-4o."""

import os
from typing import Dict, Any
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EssayAnalyzer:
    """Analyze essays using GPT-4o for feature extraction."""
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the Azure OpenAI client."""
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            self.client = None
    
    def analyze_essay(self, essay_text: str, applicant_id: str = "unknown") -> Dict[str, float]:
        """Analyze an essay and extract features.
        
        Args:
            essay_text: The essay text to analyze
            applicant_id: ID for logging purposes
            
        Returns:
            Dictionary of essay features
        """
        if not self.client:
            logger.warning("No Azure OpenAI client available, returning default features")
            return self._get_default_features()
        
        prompts = {
            'overall_essay_score': "Rate the overall quality of this medical school application essay on a scale of 0-100, considering writing quality, coherence, impact, and authenticity.",
            'motivation_authenticity': "Rate from 0-100 how genuine and personal the applicant's motivations for medicine appear to be.",
            'clinical_insight': "Rate from 0-100 the depth of clinical understanding and realistic view of medicine demonstrated.",
            'leadership_impact': "Rate from 0-100 the evidence of leadership with tangible outcomes and initiative shown.",
            'service_genuineness': "Rate from 0-100 the applicant's genuine commitment to serving others versus resume building.",
            'intellectual_curiosity': "Rate from 0-100 the evidence of intellectual engagement, research interest, and growth mindset.",
            'maturity_score': "Rate from 0-100 the emotional maturity, self-awareness, and professional readiness shown.",
            'communication_score': "Rate from 0-100 the effectiveness of written communication, clarity, and persuasiveness.",
            'diversity_contribution': "Rate from 0-100 the unique perspectives and contributions to medical school diversity offered.",
            'resilience_score': "Rate from 0-100 the evidence of resilience, perseverance, and growth from challenges.",
            'ethical_reasoning': "Rate from 0-100 the understanding of medical ethics, integrity, and professional values shown.",
            'red_flags': "Count the number of concerning elements (professionalism lapses, unrealistic views, concerning behaviors).",
            'green_flags': "Count the number of exceptional elements (outstanding achievements, exceptional insights, unique strengths)."
        }
        
        features = {}
        
        try:
            for feature, prompt in prompts.items():
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert medical school admissions evaluator. Provide only a numeric score or count as requested."
                        },
                        {
                            "role": "user", 
                            "content": f"{prompt}\n\nEssay:\n{essay_text[:4000]}"  # Limit length
                        }
                    ],
                    temperature=0.15,
                    top_p=0.9,
                    max_tokens=10
                )
                
                try:
                    value = response.choices[0].message.content.strip()
                    if 'flags' in feature:
                        features[f'llm_{feature}_count'] = int(float(value))
                    else:
                        features[f'llm_{feature}'] = float(value)
                except ValueError:
                    logger.warning(f"Could not parse response for {feature}: {value}")
                    features[f'llm_{feature}' if 'flags' not in feature else f'llm_{feature}_count'] = 0
                    
        except Exception as e:
            logger.error(f"Error analyzing essay for {applicant_id}: {e}")
            return self._get_default_features()
        
        logger.info(f"Successfully analyzed essay for {applicant_id}")
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when API is unavailable."""
        return {
            'llm_overall_essay_score': 70.0,
            'llm_motivation_authenticity': 70.0,
            'llm_clinical_insight': 70.0,
            'llm_leadership_impact': 70.0,
            'llm_service_genuineness': 70.0,
            'llm_intellectual_curiosity': 70.0,
            'llm_maturity_score': 70.0,
            'llm_communication_score': 70.0,
            'llm_diversity_contribution': 70.0,
            'llm_resilience_score': 70.0,
            'llm_ethical_reasoning': 70.0,
            'llm_red_flag_count': 0,
            'llm_green_flag_count': 1
        }