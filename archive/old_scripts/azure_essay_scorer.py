"""
Azure OpenAI Essay Scoring System for Medical Admissions
=======================================================

This module implements a specialized Azure OpenAI assistant for scoring
unstructured application data (essays, personal statements, etc.) with
consistent, rubric-based evaluation.

Key Features:
- Structured JSON output for integration with ML models
- Consistent scoring across all essays
- Multiple evaluation dimensions
- Caching to reduce API calls
- Batch processing support
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

# For local caching if Redis not available
import shelve


class AzureEssayScorer:
    """
    Azure OpenAI-based essay scoring system with optimized prompts
    and caching for medical admissions evaluation.
    """
    
    def __init__(self,
                 azure_endpoint: str,
                 api_key: str,
                 deployment_name: str = "gpt-4",
                 api_version: str = "2024-02-15-preview",
                 use_cache: bool = True,
                 cache_type: str = "local"):  # "local" or "redis"
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        
        # Model parameters optimized for consistency
        self.model_params = {
            "temperature": 0.15,  # Low for consistency, slight variation for edge cases
            "top_p": 0.9,        # Focused but not too restrictive
            "max_tokens": 800,   # Enough for detailed analysis
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Initialize caching
        self.use_cache = use_cache
        if use_cache:
            if cache_type == "redis":
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.cache_type = "redis"
            else:
                self.cache_path = Path("cache/essay_scores.db")
                self.cache_path.parent.mkdir(exist_ok=True)
                self.cache = None  # Will open on demand
                self.cache_type = "local"
        
        # System prompt for medical admissions evaluation
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the optimized system prompt for essay scoring"""
        return """You are an expert medical school admissions evaluator with 20+ years of experience. 
Your task is to evaluate applicant essays and unstructured text data according to strict medical school admission criteria.

EVALUATION FRAMEWORK:

You must score each essay on the following dimensions (1-10 scale):

1. **Motivation for Medicine** (motivation_score)
   - Clear understanding of medical profession
   - Genuine, well-articulated reasons for pursuing medicine
   - Evidence of long-term commitment
   
2. **Clinical Exposure** (clinical_understanding)
   - Depth of clinical experiences described
   - Understanding of patient care
   - Reflection on clinical observations
   
3. **Service Orientation** (service_commitment)
   - Evidence of altruism and community service
   - Understanding of physician's role in society
   - Sustained service activities
   
4. **Resilience & Growth** (resilience_score)
   - Overcoming challenges
   - Learning from failures
   - Personal growth and maturity
   
5. **Academic Preparedness** (academic_readiness)
   - Understanding of academic rigor
   - Evidence of scientific thinking
   - Intellectual curiosity
   
6. **Interpersonal Skills** (interpersonal_skills)
   - Teamwork examples
   - Communication abilities
   - Cultural competence
   
7. **Leadership Potential** (leadership_score)
   - Initiative taking
   - Influence on others
   - Vision for healthcare improvement
   
8. **Ethical Reasoning** (ethical_maturity)
   - Understanding of medical ethics
   - Moral reasoning examples
   - Professional behavior

ADDITIONAL ASSESSMENTS:

- **red_flags**: Array of specific concerns (e.g., ["Lack of clinical exposure", "Poor communication", "Ethical concerns"])
- **green_flags**: Array of exceptional qualities (e.g., ["Published research", "Extensive clinical experience", "Compelling personal narrative"])
- **overall_score**: Holistic score (1-100) weighing all factors
- **recommendation**: One of ["Strongly Recommend", "Recommend", "Neutral", "Not Recommend"]
- **summary**: 2-3 sentence evaluation summary

SCORING GUIDELINES:
- 1-3: Significantly below expectations
- 4-5: Below average
- 6-7: Average to above average
- 8-9: Excellent
- 10: Exceptional (rare)

Be objective, consistent, and base scores strictly on evidence presented in the text.
Return ONLY valid JSON with all required fields."""
    
    def _create_essay_prompt(self, essay_text: str, additional_context: Optional[Dict] = None) -> str:
        """Create the user prompt for a specific essay"""
        prompt = f"""Please evaluate the following medical school application essay:

ESSAY TEXT:
---
{essay_text[:4000]}  # Truncate to manage tokens
---"""

        if additional_context:
            prompt += f"""

ADDITIONAL CONTEXT:
- Applicant Age: {additional_context.get('age', 'Unknown')}
- Total Experience Hours: {additional_context.get('total_hours', 'Unknown')}
- Research Hours: {additional_context.get('research_hours', 'Unknown')}
- Clinical Hours: {additional_context.get('clinical_hours', 'Unknown')}
- GPA Trend: {additional_context.get('gpa_trend', 'Unknown')}"""

        prompt += """

Provide your evaluation as a JSON object with all required scores and assessments."""
        
        return prompt
    
    def _get_cache_key(self, text: str, context: Optional[Dict] = None) -> str:
        """Generate cache key from text and context"""
        cache_data = text + str(context) if context else text
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Retrieve result from cache"""
        if not self.use_cache:
            return None
            
        try:
            if self.cache_type == "redis":
                cached = self.cache.get(cache_key)
                return json.loads(cached) if cached else None
            else:
                with shelve.open(str(self.cache_path)) as cache:
                    return cache.get(cache_key)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save result to cache"""
        if not self.use_cache:
            return
            
        try:
            if self.cache_type == "redis":
                self.cache.setex(
                    cache_key,
                    86400 * 7,  # 7 days TTL
                    json.dumps(result)
                )
            else:
                with shelve.open(str(self.cache_path)) as cache:
                    cache[cache_key] = result
        except Exception as e:
            print(f"Cache save error: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def score_essay_async(self, 
                               essay_text: str, 
                               additional_context: Optional[Dict] = None) -> Dict:
        """Score a single essay asynchronously with retry logic"""
        
        # Check cache first
        cache_key = self._get_cache_key(essay_text, additional_context)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_essay_prompt(essay_text, additional_context)}
        ]
        
        try:
            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.deployment_name,
                messages=messages,
                response_format={"type": "json_object"},
                **self.model_params
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Validate required fields
            required_fields = [
                'motivation_score', 'clinical_understanding', 'service_commitment',
                'resilience_score', 'academic_readiness', 'interpersonal_skills',
                'leadership_score', 'ethical_maturity', 'red_flags', 'green_flags',
                'overall_score', 'recommendation', 'summary'
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add metadata
            result['scored_at'] = datetime.now().isoformat()
            result['model_used'] = self.deployment_name
            result['from_cache'] = False
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"Error scoring essay: {e}")
            # Return default scores on error
            return self._get_default_scores()
    
    def score_essay(self, essay_text: str, additional_context: Optional[Dict] = None) -> Dict:
        """Synchronous wrapper for essay scoring"""
        return asyncio.run(self.score_essay_async(essay_text, additional_context))
    
    def _get_default_scores(self) -> Dict:
        """Return default scores when API fails"""
        return {
            'motivation_score': 5,
            'clinical_understanding': 5,
            'service_commitment': 5,
            'resilience_score': 5,
            'academic_readiness': 5,
            'interpersonal_skills': 5,
            'leadership_score': 5,
            'ethical_maturity': 5,
            'red_flags': ['Unable to score - API error'],
            'green_flags': [],
            'overall_score': 50,
            'recommendation': 'Neutral',
            'summary': 'Unable to evaluate due to technical error.',
            'scored_at': datetime.now().isoformat(),
            'from_cache': False,
            'error': True
        }
    
    async def batch_score_essays(self, 
                                essays: List[Tuple[str, str, Optional[Dict]]],
                                batch_size: int = 10) -> List[Dict]:
        """
        Score multiple essays in batches
        
        Args:
            essays: List of tuples (applicant_id, essay_text, optional_context)
            batch_size: Number of concurrent API calls
        
        Returns:
            List of scoring results with applicant_id included
        """
        results = []
        
        for i in range(0, len(essays), batch_size):
            batch = essays[i:i + batch_size]
            
            # Create tasks for batch
            tasks = []
            for applicant_id, essay_text, context in batch:
                task = self.score_essay_async(essay_text, context)
                tasks.append((applicant_id, task))
            
            # Execute batch
            batch_results = []
            for applicant_id, task in tasks:
                result = await task
                result['applicant_id'] = applicant_id
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(essays):
                await asyncio.sleep(1)
        
        return results
    
    def create_scoring_report(self, results: List[Dict]) -> pd.DataFrame:
        """Create a summary report from scoring results"""
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add derived metrics
        df['avg_dimension_score'] = df[[
            'motivation_score', 'clinical_understanding', 'service_commitment',
            'resilience_score', 'academic_readiness', 'interpersonal_skills',
            'leadership_score', 'ethical_maturity'
        ]].mean(axis=1)
        
        df['has_red_flags'] = df['red_flags'].apply(lambda x: len(x) > 0)
        df['num_green_flags'] = df['green_flags'].apply(len)
        
        # Recommendation encoding
        rec_map = {
            'Strongly Recommend': 4,
            'Recommend': 3,
            'Neutral': 2,
            'Not Recommend': 1
        }
        df['recommendation_score'] = df['recommendation'].map(rec_map)
        
        return df


# Utility functions for integration
def create_azure_config() -> Dict:
    """Create Azure configuration from environment or defaults"""
    return {
        'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT', 'https://your-resource.openai.azure.com/'),
        'api_key': os.getenv('AZURE_OPENAI_KEY', 'your-api-key'),
        'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4'),
        'api_version': '2024-02-15-preview'
    }


def integrate_essay_scores_with_model(essay_scores_df: pd.DataFrame, 
                                    applicant_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate essay scores with structured features for final model
    
    Args:
        essay_scores_df: DataFrame with essay scoring results
        applicant_features_df: DataFrame with structured applicant features
    
    Returns:
        Combined DataFrame ready for final prediction model
    """
    
    # Merge on applicant_id
    combined = applicant_features_df.merge(
        essay_scores_df, 
        left_on='AMCAS_ID', 
        right_on='applicant_id', 
        how='left'
    )
    
    # Fill missing essay scores with defaults
    essay_score_cols = [
        'motivation_score', 'clinical_understanding', 'service_commitment',
        'resilience_score', 'academic_readiness', 'interpersonal_skills',
        'leadership_score', 'ethical_maturity', 'overall_score'
    ]
    
    for col in essay_score_cols:
        combined[col] = combined[col].fillna(5)  # Neutral score
    
    combined['recommendation_score'] = combined['recommendation_score'].fillna(2)  # Neutral
    combined['has_red_flags'] = combined['has_red_flags'].fillna(False)
    combined['num_green_flags'] = combined['num_green_flags'].fillna(0)
    
    return combined


# Example usage
async def main():
    """Example of using the Azure Essay Scorer"""
    
    # Initialize scorer
    config = create_azure_config()
    scorer = AzureEssayScorer(**config)
    
    # Example essays to score
    essays = [
        (
            "12345",
            "My journey to medicine began when I volunteered at a local hospital...",
            {"age": 24, "total_hours": 2000, "research_hours": 500}
        ),
        (
            "12346", 
            "Growing up in an underserved community, I witnessed firsthand...",
            {"age": 26, "total_hours": 1500, "clinical_hours": 800}
        )
    ]
    
    # Score essays
    results = await scorer.batch_score_essays(essays)
    
    # Create report
    report_df = scorer.create_scoring_report(results)
    print(report_df)
    
    # Save results
    report_df.to_csv("essay_scores.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())