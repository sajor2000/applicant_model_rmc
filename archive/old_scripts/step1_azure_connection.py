"""
Step 1: Azure OpenAI Connection Setup
=====================================

This script establishes a secure connection to Azure OpenAI
and provides methods for calling the API with retry logic.
"""

import os
from openai import AzureOpenAI
from typing import Dict, Optional
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureOpenAIConnection:
    """
    Manages connection to Azure OpenAI for medical admissions essay evaluation
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 deployment_name: Optional[str] = None,
                 api_version: str = "2025-01-01-preview"):
        """
        Initialize Azure OpenAI connection
        
        Args:
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)
            endpoint: Azure endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)
            deployment_name: Model deployment name (or set AZURE_OPENAI_DEPLOYMENT env var)
            api_version: API version to use
        """
        
        # Get credentials from parameters or environment variables
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable.")
        
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint
        )
        
        # Model parameters for consistent scoring
        self.model_params = {
            'temperature': 0.15,  # Low for consistency
            'top_p': 0.9,        # Focused but not too restrictive
            'max_tokens': 500,   # Enough for JSON response
            'response_format': {"type": "json_object"}
        }
        
        logger.info(f"Azure OpenAI connection initialized with deployment: {self.deployment_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def evaluate_essays(self, 
                       personal_statement: str,
                       secondary_essays: Dict[str, str],
                       experiences: str,
                       system_prompt: str,
                       user_prompt_template: str) -> Dict:
        """
        Send essays to Azure OpenAI for evaluation
        
        Args:
            personal_statement: Main medical school essay
            secondary_essays: Dictionary of secondary essay responses
            experiences: Formatted experience descriptions
            system_prompt: System instructions for the model
            user_prompt_template: Template for formatting the essays
            
        Returns:
            Dictionary with numeric scores
        """
        
        # Format the user prompt with the actual essay content
        user_prompt = user_prompt_template.format(
            personal_statement=personal_statement[:3500],  # Limit length
            secondary_essays=self._format_secondary_essays(secondary_essays),
            experiences=experiences[:2000]  # Limit length
        )
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.model_params
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            scores = json.loads(response_text)
            
            # Validate response has all required fields
            required_fields = [
                'llm_narrative_coherence', 'llm_motivation_authenticity',
                'llm_reflection_depth', 'llm_growth_demonstrated',
                'llm_unique_perspective', 'llm_clinical_insight',
                'llm_service_genuineness', 'llm_leadership_impact',
                'llm_communication_quality', 'llm_maturity_score',
                'llm_red_flag_count', 'llm_green_flag_count',
                'llm_overall_essay_score'
            ]
            
            for field in required_fields:
                if field not in scores:
                    logger.warning(f"Missing field in response: {field}")
                    scores[field] = 5 if field.endswith('_score') else 0
            
            return scores
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._get_default_scores()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _format_secondary_essays(self, essays: Dict[str, str]) -> str:
        """Format secondary essays for the prompt"""
        formatted = ""
        
        # Key essays to include
        essay_keys = [
            ('1 - Personal Attributes / Life Experiences', 'Personal Attributes'),
            ('2 - Challenging Situation', 'Challenge Response'),
            ('3 - Reflect Experience', 'Experience Reflection'),
            ('4 - Hope to Gain', 'Goals'),
            ('6 - Experiences', 'Additional Experiences'),
            ('7 - COVID Impact', 'COVID Impact')
        ]
        
        for full_key, short_name in essay_keys:
            if full_key in essays and essays[full_key]:
                formatted += f"\n{short_name}:\n{essays[full_key][:500]}\n"
        
        return formatted
    
    def _get_default_scores(self) -> Dict:
        """Return neutral scores if evaluation fails"""
        return {
            'llm_narrative_coherence': 5,
            'llm_motivation_authenticity': 5,
            'llm_reflection_depth': 5,
            'llm_growth_demonstrated': 5,
            'llm_unique_perspective': 5,
            'llm_clinical_insight': 5,
            'llm_service_genuineness': 5,
            'llm_leadership_impact': 5,
            'llm_communication_quality': 5,
            'llm_maturity_score': 5,
            'llm_red_flag_count': 0,
            'llm_green_flag_count': 0,
            'llm_overall_essay_score': 50
        }
    
    def test_connection(self) -> bool:
        """Test the Azure OpenAI connection with a simple prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with 'Connection successful' if you receive this."}
                ],
                max_tokens=50
            )
            
            result = response.choices[0].message.content
            logger.info(f"Connection test result: {result}")
            return "successful" in result.lower()
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Example usage and setup instructions
if __name__ == "__main__":
    print("Azure OpenAI Connection Setup")
    print("="*50)
    print("\nTo use this script, set the following environment variables:")
    print("1. AZURE_OPENAI_API_KEY - Your Azure OpenAI API key")
    print("2. AZURE_OPENAI_ENDPOINT - Your endpoint URL (e.g., https://your-resource.openai.azure.com/)")
    print("3. AZURE_OPENAI_DEPLOYMENT - Your model deployment name (e.g., gpt-4)")
    print("\nExample:")
    print('export AZURE_OPENAI_API_KEY="your-key-here"')
    print('export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"')
    print('export AZURE_OPENAI_DEPLOYMENT="gpt-4"')
    
    # Test connection if credentials are available
    try:
        azure_client = AzureOpenAIConnection()
        if azure_client.test_connection():
            print("\n✅ Connection successful!")
        else:
            print("\n❌ Connection failed!")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")