# Integration Architecture: Combining LLM and ML Outputs

## Overview

The LLM outputs numeric scores (not text) that become features for the final ML classifier alongside structured data.

## Data Flow Architecture

```
Applicant Data
    ├── Structured Data (Excel files 1-12)
    │   ├── Age, GPA, MCAT scores
    │   ├── Experience hours
    │   ├── Service ratings
    │   └── Academic metrics
    │         ↓
    │   [ML Model Features]
    │         ↓
    └── Unstructured Data (Files 6, 9, 10)
        ├── Personal statement
        ├── Secondary essays
        └── Experience descriptions
              ↓
        [Azure LLM Scoring]
              ↓
        [Numeric Scores Only]
              ↓
    [Combined Feature Vector]
              ↓
    [Final ML Classifier]
              ↓
    [Tier Prediction 1-4]
```

## LLM Output Structure (Numeric Only)

The LLM must output **numeric scores** that can be used as features:

```json
{
  "llm_motivation_score": 8.5,           // 0-10 scale
  "llm_clinical_understanding": 7.0,     // 0-10 scale
  "llm_service_commitment": 8.0,         // 0-10 scale
  "llm_resilience_score": 9.0,           // 0-10 scale
  "llm_academic_readiness": 7.5,         // 0-10 scale
  "llm_interpersonal_skills": 8.0,       // 0-10 scale
  "llm_leadership_score": 6.5,           // 0-10 scale
  "llm_ethical_maturity": 7.0,           // 0-10 scale
  
  "llm_overall_score": 76.5,             // 0-100 scale
  "llm_tier_prediction": 3,              // 1-4 (numeric)
  "llm_interview_recommend": 1,          // Binary: 0=No, 1=Yes
  
  "llm_red_flag_count": 0,               // Count of concerns
  "llm_green_flag_count": 2,             // Count of strengths
  "llm_confidence_score": 0.85           // 0-1 confidence
}
```

## Combined Feature Vector for Final ML Model

```python
# Example of how features are combined
def create_combined_features(structured_data, llm_scores):
    """
    Combines structured and LLM features for final ML model
    """
    combined_features = {
        # Structured features (from Excel files)
        'age': structured_data['Age'],
        'gpa_total': structured_data['Total_GPA'],
        'mcat_total': structured_data['MCAT_Total_Score'],
        'exp_hour_total': structured_data['Exp_Hour_Total'],
        'exp_hour_research': structured_data['Exp_Hour_Research'],
        'exp_hour_clinical': structured_data['Exp_Hour_Volunteer_Med'],
        'service_rating': structured_data['Service Rating (Numerical)'],
        'gpa_trend': structured_data['Total_GPA_Trend'],
        'disadvantaged_ind': structured_data['Disadvantanged_Ind'],
        
        # Engineered structured features
        'research_intensity': calculate_research_intensity(structured_data),
        'clinical_intensity': calculate_clinical_intensity(structured_data),
        'experience_balance': calculate_experience_balance(structured_data),
        
        # LLM-derived features (all numeric)
        'llm_motivation': llm_scores['llm_motivation_score'],
        'llm_clinical_insight': llm_scores['llm_clinical_understanding'],
        'llm_service': llm_scores['llm_service_commitment'],
        'llm_resilience': llm_scores['llm_resilience_score'],
        'llm_academic': llm_scores['llm_academic_readiness'],
        'llm_interpersonal': llm_scores['llm_interpersonal_skills'],
        'llm_leadership': llm_scores['llm_leadership_score'],
        'llm_ethics': llm_scores['llm_ethical_maturity'],
        'llm_overall': llm_scores['llm_overall_score'] / 10,  # Normalize to 0-10
        'llm_confidence': llm_scores['llm_confidence_score'],
        
        # Interaction features (structured × LLM)
        'experience_quality': structured_data['Exp_Hour_Total'] * llm_scores['llm_clinical_understanding'] / 10,
        'service_impact': structured_data['Service Rating (Numerical)'] * llm_scores['llm_service_commitment'] / 10,
        'academic_potential': structured_data['Total_GPA'] * llm_scores['llm_academic_readiness'] / 10
    }
    
    return combined_features
```

## Final ML Classifier Design

```python
class IntegratedAdmissionsClassifier:
    """
    Final classifier combining structured and LLM features
    """
    
    def __init__(self):
        # Feature names in order
        self.feature_names = [
            # Structured features
            'age', 'gpa_total', 'mcat_total', 'exp_hour_total',
            'exp_hour_research', 'exp_hour_clinical', 'service_rating',
            'gpa_trend', 'disadvantaged_ind', 'research_intensity',
            'clinical_intensity', 'experience_balance',
            
            # LLM features
            'llm_motivation', 'llm_clinical_insight', 'llm_service',
            'llm_resilience', 'llm_academic', 'llm_interpersonal',
            'llm_leadership', 'llm_ethics', 'llm_overall', 'llm_confidence',
            
            # Interaction features
            'experience_quality', 'service_impact', 'academic_potential'
        ]
        
        # Model (trained on 2022-2023 data)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced'
        )
        
        # Feature importance tracking
        self.feature_importance = {}
    
    def predict(self, structured_data, llm_scores):
        # Create combined feature vector
        features = create_combined_features(structured_data, llm_scores)
        
        # Convert to array in correct order
        X = np.array([features[name] for name in self.feature_names])
        
        # Make prediction
        tier_prediction = self.model.predict([X])[0]
        tier_probability = self.model.predict_proba([X])[0]
        
        return {
            'final_tier': tier_prediction,
            'tier_probabilities': tier_probability,
            'confidence': tier_probability.max()
        }
```

## Why Text Outputs Don't Work

You're correct that text outputs like "key_strengths" and "areas_concern" cannot be directly used in an ML classifier. Here's why:

1. **ML models need numeric features**
   - Text would need embedding/encoding
   - Loses interpretability
   - Adds complexity

2. **Instead, we convert insights to numbers**:
   - Count of strengths → `llm_green_flag_count`
   - Severity of concerns → `llm_red_flag_count`
   - Quality assessments → numeric scores

3. **Text outputs are kept for human review**:
   - Stored separately for committee discussion
   - Used to explain decisions
   - Not part of ML pipeline

## Integration Weights

The final model learns optimal weights, but typically:

```python
# Approximate feature importance
structured_weight = 0.65  # 65% from traditional metrics
llm_weight = 0.35        # 35% from essay analysis

# But the ML model determines actual weights through training
```

## Example End-to-End Flow

```python
def evaluate_applicant_complete(applicant_id):
    # 1. Load structured data
    structured = load_structured_data(applicant_id)
    
    # 2. Load unstructured text
    essays = load_unstructured_content(applicant_id)
    
    # 3. Get LLM scores (numeric only)
    llm_scores = azure_llm_evaluate(essays)  # Returns numbers
    
    # 4. Combine features
    combined_features = create_combined_features(structured, llm_scores)
    
    # 5. Final ML prediction
    final_prediction = integrated_model.predict(combined_features)
    
    # 6. Return results
    return {
        'applicant_id': applicant_id,
        'final_tier': final_prediction['final_tier'],
        'confidence': final_prediction['confidence'],
        'structured_contribution': structured_features_impact,
        'llm_contribution': llm_features_impact,
        
        # Keep text for human review (not used in ML)
        'human_readable': {
            'strengths': llm_scores['key_strengths_text'],
            'concerns': llm_scores['areas_concern_text'],
            'summary': llm_scores['summary_text']
        }
    }
```

## Key Insights

1. **LLM outputs must be numeric** for ML integration
2. **Text insights** are stored separately for human review
3. **Combined features** include both structured and LLM scores
4. **Interaction features** capture synergies between metrics
5. **Final model** learns optimal weights through training

This architecture ensures:
- Clean integration between LLM and ML
- Interpretable features
- Fair evaluation
- Human-readable explanations alongside ML predictions