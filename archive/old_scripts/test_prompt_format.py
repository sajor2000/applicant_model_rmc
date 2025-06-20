"""
Test the prompt format with Azure OpenAI
========================================
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import json

# Load environment variables
load_dotenv()

# Initialize client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview"
)

# Test with the medical admissions prompt
system_prompt = """You are an expert medical school admissions reviewer. Evaluate the applicant and return ONLY a JSON object with these exact numeric fields:
- llm_narrative_coherence (0-10)
- llm_motivation_authenticity (0-10)
- llm_reflection_depth (0-10)
- llm_growth_demonstrated (0-10)
- llm_unique_perspective (0-10)
- llm_clinical_insight (0-10)
- llm_service_genuineness (0-10)
- llm_leadership_impact (0-10)
- llm_communication_quality (0-10)
- llm_maturity_score (0-10)
- llm_red_flag_count (integer 0-10)
- llm_green_flag_count (integer 0-10)
- llm_overall_essay_score (0-100)"""

user_prompt = """Evaluate this medical school application:

PERSONAL STATEMENT:
My journey to medicine began when my grandmother was diagnosed with Alzheimer's disease. Spending time with her at the care facility, I witnessed firsthand the profound impact that compassionate healthcare providers can have on patients and families. This experience inspired me to volunteer at the facility, where I spent over 500 hours developing a music therapy program that now serves 50 residents weekly.

SECONDARY ESSAY - Challenge:
During COVID-19, our research lab shut down. I took initiative to adapt our protocol for remote data collection, coordinating 12 team members across different time zones. This resulted in our continued progress and eventual publication in the Journal of Undergraduate Research.

EXPERIENCES:
- Research Assistant: 800 hours studying Alzheimer's biomarkers
- Hospital Volunteer: 600 hours in the emergency department
- Music Therapy Program Founder: Created and led program for memory care patients"""

print("Testing prompt with Azure OpenAI GPT-4o...")

try:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.15,
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    print("\nRaw response:")
    print(result)
    
    # Parse JSON
    scores = json.loads(result)
    print("\nParsed scores:")
    for key, value in scores.items():
        print(f"  {key}: {value}")
        
except Exception as e:
    print(f"Error: {e}")
    
    # Try without JSON mode
    print("\nTrying without JSON response format...")
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.15
    )
    print("Response:", response.choices[0].message.content)