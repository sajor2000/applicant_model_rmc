# Systematic Framework for Fair and Equitable Medical Admissions Evaluation

## Overview

Based on analysis of the available data files, this framework defines which unstructured text should be evaluated by the LLM and how to ensure fair, unbiased assessment.

## Files for LLM Analysis

### Primary Sources (High Priority)

1. **Personal Statement** (`9. Personal Statement.xlsx`)
   - **Content**: Full personal statements (~5,000 characters)
   - **Why Include**: Core narrative revealing motivation, experiences, and reflection
   - **What to Extract**: Medical motivation, personal growth, unique perspectives

2. **Secondary Application Essays** (`10. Secondary Application.xlsx`)
   - **Content**: 6-7 essay responses (~900-1,100 characters each)
   - **Why Include**: Targeted responses showing depth and consistency
   - **Key Essays**:
     - Personal Attributes/Life Experiences
     - Challenging Situation Response
     - Reflection on Experiences
     - Learning Goals
     - COVID Impact (recent context)

3. **Experience Descriptions** (`6. Experiences.xlsx`)
   - **Content**: Activity descriptions and meaningful experience reflections
   - **Why Include**: Shows depth of involvement and impact
   - **What to Extract**: Leadership, initiative, understanding of activities

### Conditional Sources (Context-Dependent)

4. **Hardship Comments** (`1. Applicants.xlsx` - selective field)
   - **Content**: Explanations of challenges/disadvantages (~1,200 characters)
   - **Why Include**: Critical context for achievement evaluation
   - **Caution**: Use only to understand obstacles overcome, not to penalize

5. **Institutional Action Descriptions** (`1. Applicants.xlsx` - if present)
   - **Content**: Explanations of academic/conduct issues
   - **Why Include**: Context for any red flags
   - **Approach**: Focus on growth and learning demonstrated

## Systematic Evaluation Process

### Step 1: Data Aggregation
```python
def aggregate_unstructured_content(applicant_id):
    content = {
        "personal_statement": load_from_file_9(applicant_id),
        "secondary_essays": {
            "attributes": load_essay_1_from_file_10(applicant_id),
            "challenge": load_essay_2_from_file_10(applicant_id),
            "reflection": load_essay_3_from_file_10(applicant_id),
            "goals": load_essay_4_from_file_10(applicant_id),
            "experiences": load_essay_6_from_file_10(applicant_id),
            "covid_impact": load_essay_7_from_file_10(applicant_id)
        },
        "experience_descriptions": load_from_file_6(applicant_id),
        "hardship_context": load_if_present_from_file_1(applicant_id)
    }
    return content
```

### Step 2: Structured LLM Evaluation

The LLM evaluates based on **content and quality** only, not demographics:

```json
{
  "content_evaluation": {
    "motivation_authenticity": "Score based on depth of reflection",
    "experience_quality": "Score based on impact and learning",
    "communication_effectiveness": "Score based on clarity and coherence",
    "growth_mindset": "Score based on learning from challenges",
    "service_orientation": "Score based on commitment evidence",
    "professional_readiness": "Score based on understanding of medicine"
  },
  "red_flags": "Content-based concerns only",
  "green_flags": "Exceptional qualities demonstrated"
}
```

### Step 3: Context-Aware Scoring

When evaluating achievements, consider context WITHOUT introducing bias:

```python
def evaluate_in_context(achievements, hardship_described):
    """
    Evaluate achievements relative to opportunities available,
    not absolute metrics
    """
    if hardship_described:
        # Recognize achievement despite obstacles
        # Do NOT lower standards, but acknowledge context
        context_factor = assess_obstacle_severity(hardship_described)
    else:
        context_factor = "standard_context"
    
    return adjusted_evaluation
```

## Bias Prevention Strategies

### 1. Exclude Demographic Identifiers from LLM

**DO NOT** send to LLM:
- Names, gender, age
- Race, ethnicity, citizenship
- Family income, parental education
- Geographic location
- School names (unless directly relevant to experience)

### 2. Standardize Evaluation Criteria

Focus exclusively on:
- **Quality of reflection** (not quantity of experiences)
- **Impact created** (relative to opportunities)
- **Growth demonstrated** (not starting point)
- **Understanding shown** (not prestigious names)
- **Commitment evidence** (sustained involvement)

### 3. Mitigate Linguistic Bias

Instruct LLM to:
- Value clarity over sophisticated vocabulary
- Recognize diverse communication styles
- Focus on content, not writing style
- Consider English as second language without penalty

### 4. Systematic Prompt Structure

```python
BIAS_PREVENTION_PROMPT = """
Evaluate based ONLY on:
1. Evidence of commitment to medicine
2. Quality of experiences and reflection
3. Personal growth and resilience
4. Service to others
5. Understanding of healthcare

DO NOT consider or infer:
- Applicant demographics
- Socioeconomic background
- Institution prestige
- Geographic location
- Cultural background

Focus on WHAT they did and LEARNED, not WHERE or WITH WHOM.
"""
```

## Quality Assurance Framework

### 1. Consistency Checks
- Regular calibration with test cases
- Score distribution monitoring
- Inter-rater reliability (if multiple LLM calls)

### 2. Bias Auditing
```python
def audit_for_bias(evaluation_results):
    # Check score distributions across:
    # - Essay length (shouldn't favor longer)
    # - Vocabulary complexity (shouldn't favor sophisticated)
    # - Experience types (shouldn't favor traditional paths)
    
    # Flag potential bias indicators:
    # - Unusual score clustering
    # - Correlation with text features vs content
    # - Systematic differences in similar content
```

### 3. Transparency Requirements

Each evaluation must include:
- Specific evidence cited for scores
- Clear reasoning for tier placement
- Explicit factors for interview decision

## Implementation Example

```python
def evaluate_applicant_fairly(applicant_id):
    # 1. Load unstructured content
    content = aggregate_unstructured_content(applicant_id)
    
    # 2. Anonymize and prepare for LLM
    anonymous_content = remove_identifying_information(content)
    
    # 3. Apply systematic evaluation
    evaluation = llm_evaluate(
        content=anonymous_content,
        prompt=BIAS_PREVENTION_PROMPT,
        rubric=STANDARDIZED_RUBRIC
    )
    
    # 4. Quality check
    evaluation = validate_evaluation_fairness(evaluation)
    
    # 5. Combine with structured data
    final_score = integrate_evaluations(
        llm_eval=evaluation,
        ml_prediction=structured_model_output,
        weights={"llm": 0.35, "ml": 0.65}
    )
    
    return final_score
```

## Files NOT Used for LLM Evaluation

These contain structured data for ML model only:
- Academic Records (grades)
- GPA Trends
- Language proficiency
- Family information
- School details

These files are intentionally excluded from LLM evaluation to prevent bias based on:
- Institution prestige
- Family educational background
- Language spoken at home
- Geographic indicators

## Monitoring and Adjustment

1. **Regular Review**: Monthly analysis of score distributions
2. **Feedback Loop**: Compare LLM evaluations with final admission outcomes
3. **Continuous Improvement**: Refine prompts based on identified biases
4. **Documentation**: Track all changes to maintain consistency

## Ethical Principles

1. **Equity**: Evaluate achievement relative to opportunity
2. **Fairness**: Apply same standards to all applicants
3. **Transparency**: Clear criteria and reasoning
4. **Accountability**: Regular bias auditing
5. **Inclusivity**: Value diverse paths to medicine

This framework ensures systematic, fair evaluation while leveraging the LLM's ability to understand nuanced narratives and assess qualities that numbers alone cannot capture.