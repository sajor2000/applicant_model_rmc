"""
Optimized Azure OpenAI Prompt for Medical Admissions
===================================================

This module contains the refined prompt specifically designed to distinguish
between Tier 2 (Potential Review) and Tier 3 (Probable Interview) candidates.
"""

SYSTEM_PROMPT_OPTIMIZED = """You are an expert medical school admissions evaluator with over 20 years of experience. Your PRIMARY RESPONSIBILITY is to accurately distinguish between candidates who should receive interviews (Tiers 3-4) versus those who should not (Tiers 1-2).

CRITICAL FOCUS: The distinction between Tier 2 (Potential Review) and Tier 3 (Probable Interview) is your most important decision. Many qualified candidates fall in this borderline zone.

TIER DEFINITIONS WITH CLEAR BOUNDARIES:

Tier 4 - Very Likely Interview (85-100):
- Exceptional candidates with multiple standout qualities
- Would be a significant loss to not interview

Tier 3 - Probable Interview (70-84):
- DEMONSTRATES READINESS for medical school
- Has at least ONE distinguishing strength
- Shows genuine understanding of medicine through experience
- Clear upward trajectory and growth

Tier 2 - Potential Review (50-69):
- LACKS CRITICAL READINESS ELEMENTS
- Basic qualifications but no distinguishing features
- Limited clinical exposure OR poor reflection
- Would need committee debate to justify interview

Tier 1 - Very Unlikely (0-49):
- Significant deficiencies or red flags
- Not ready for medical school

KEY DIFFERENTIATORS for Tier 2 vs 3:
✓ Quality of clinical reflection (not just hours)
✓ Evidence of impact or initiative
✓ Depth of understanding about medicine
✓ Personal growth narrative
✓ Unique contribution to class

Evaluate these dimensions (1-10):
- motivation_score: Depth and authenticity of medical calling
- clinical_understanding: Insight from clinical experiences
- service_commitment: Sustained altruistic activities
- resilience_score: Overcoming challenges with growth
- academic_readiness: Intellectual preparation
- interpersonal_skills: Communication and teamwork
- leadership_score: Initiative and impact
- ethical_maturity: Professional reasoning

Return JSON with all fields including overall_score, predicted_tier, interview_recommendation (YES/NO), and detailed justification for borderline cases."""

def create_evaluation_prompt(essay_text: str, secondary_essays: dict = None, experiences: str = None) -> str:
    """
    Create optimized prompt for specific applicant evaluation
    
    Args:
        essay_text: Primary personal statement
        secondary_essays: Dict of secondary essay responses
        experiences: Description of activities/experiences
    """
    
    prompt = f"""Evaluate this medical school applicant, paying special attention to whether they meet the threshold for interview (Tier 3+) or fall short (Tier 2).

PERSONAL STATEMENT:
{essay_text[:3500]}

"""
    
    if secondary_essays:
        prompt += "SECONDARY ESSAYS:\n"
        for question, response in list(secondary_essays.items())[:3]:  # Limit to 3
            prompt += f"\nQ: {question}\nA: {response[:500]}\n"
    
    if experiences:
        prompt += f"\nKEY EXPERIENCES:\n{experiences[:1000]}\n"
    
    prompt += """
EVALUATION TASK:
1. Score each dimension (1-10)
2. Calculate overall_score (0-100)
3. Assign predicted_tier (1-4)
4. Determine interview_recommendation (YES/NO)
5. For scores 65-75, provide detailed borderline_justification

Critical question for Tier 2/3 decision: "Would we regret not interviewing this person?"

Return complete JSON evaluation."""
    
    return prompt


# Refined scoring rubric for consistency
SCORING_RUBRIC = {
    "motivation_score": {
        "10": "Life-changing personal experience driving deep commitment",
        "9": "Profound understanding with unique perspective",
        "8": "Clear, authentic motivation with personal connection",
        "7": "Good articulation of reasons, some personal elements",
        "6": "Standard pre-med narrative, genuine but not distinctive",
        "5": "Generic reasons, limited personal connection",
        "4": "Vague or primarily external motivations",
        "3": "Unclear why medicine specifically",
        "2": "Concerning motivations (prestige, money)",
        "1": "No clear motivation evident"
    },
    
    "clinical_understanding": {
        "10": "Extensive experience with sophisticated insights",
        "9": "Deep understanding from varied clinical settings",
        "8": "Good insights from solid clinical exposure",
        "7": "Demonstrates understanding beyond observation",
        "6": "Basic understanding from moderate exposure",
        "5": "Limited exposure but shows some insight",
        "4": "Minimal exposure, surface-level understanding",
        "3": "Very limited clinical contact",
        "2": "No meaningful clinical exposure",
        "1": "Misunderstands medical practice"
    },
    
    "tier_decision_matrix": {
        "upgrade_to_3": [
            "Shows initiative creating programs/solutions",
            "Demonstrates leadership with measurable impact",
            "Unique background that enriches class diversity",
            "Research with genuine understanding/contribution",
            "Overcame significant challenges with growth",
            "Sustained commitment with deepening involvement",
            "Clear vision connecting past to future goals"
        ],
        "keep_at_2": [
            "Checkbox activities without depth",
            "Clinical hours without meaningful reflection",
            "Leadership titles without demonstrated impact",
            "Generic essays lacking authenticity",
            "No clear distinguishing characteristics",
            "Recent start to medical interests",
            "Passive participant in experiences"
        ]
    }
}


def evaluate_borderline_case(scores: dict, narrative_quality: str, unique_factors: list) -> dict:
    """
    Special handling for borderline cases (65-75 overall score)
    
    Returns:
        Dictionary with tier decision and detailed justification
    """
    
    # Calculate weighted factors
    clinical_weight = 0.25
    leadership_weight = 0.20
    uniqueness_weight = 0.20
    narrative_weight = 0.20
    trajectory_weight = 0.15
    
    upgrade_score = 0
    
    # Clinical reflection quality
    if scores['clinical_understanding'] >= 7:
        upgrade_score += clinical_weight
    
    # Leadership impact
    if scores['leadership_score'] >= 7:
        upgrade_score += leadership_weight
    
    # Unique contributions
    if len(unique_factors) >= 2:
        upgrade_score += uniqueness_weight
    
    # Narrative quality
    if narrative_quality in ['compelling', 'excellent']:
        upgrade_score += narrative_weight
    
    # Growth trajectory (resilience + academic readiness)
    if (scores['resilience_score'] + scores['academic_readiness']) / 2 >= 7:
        upgrade_score += trajectory_weight
    
    # Decision threshold
    if upgrade_score >= 0.5:
        return {
            "decision": "UPGRADE_TO_TIER_3",
            "justification": f"Upgrade factors ({upgrade_score:.2f}) exceed threshold. Strong in clinical reflection, leadership, and unique contributions.",
            "confidence": "MODERATE_HIGH"
        }
    else:
        return {
            "decision": "MAINTAIN_TIER_2",
            "justification": f"Upgrade factors ({upgrade_score:.2f}) below threshold. Lacks distinguishing elements for interview priority.",
            "confidence": "MODERATE"
        }


# Example usage for Azure OpenAI
def create_azure_assistant_config():
    """Configuration for Azure OpenAI Assistant API"""
    return {
        "name": "Medical Admissions Evaluator",
        "instructions": SYSTEM_PROMPT_OPTIMIZED,
        "model": "gpt-4",
        "temperature": 0.15,
        "top_p": 0.9,
        "response_format": {"type": "json_object"},
        "metadata": {
            "version": "2.0",
            "focus": "tier_2_3_distinction",
            "last_updated": "2024"
        }
    }


# Validation schema for output
OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "motivation_score", "clinical_understanding", "service_commitment",
        "resilience_score", "academic_readiness", "interpersonal_skills",
        "leadership_score", "ethical_maturity", "overall_score",
        "predicted_tier", "tier_name", "interview_recommendation",
        "confidence_level", "key_strengths", "red_flags", "green_flags",
        "summary"
    ],
    "properties": {
        "motivation_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "clinical_understanding": {"type": "integer", "minimum": 1, "maximum": 10},
        "service_commitment": {"type": "integer", "minimum": 1, "maximum": 10},
        "resilience_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "academic_readiness": {"type": "integer", "minimum": 1, "maximum": 10},
        "interpersonal_skills": {"type": "integer", "minimum": 1, "maximum": 10},
        "leadership_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "ethical_maturity": {"type": "integer", "minimum": 1, "maximum": 10},
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "predicted_tier": {"type": "integer", "minimum": 1, "maximum": 4},
        "tier_name": {"type": "string", "enum": ["Very Unlikely", "Potential Review", "Probable Interview", "Very Likely Interview"]},
        "interview_recommendation": {"type": "string", "enum": ["YES", "NO"]},
        "confidence_level": {"type": "string", "enum": ["LOW", "MODERATE", "MODERATE_HIGH", "HIGH"]},
        "key_strengths": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "green_flags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "borderline_justification": {"type": "string"},
        "committee_notes": {"type": "string"}
    }
}