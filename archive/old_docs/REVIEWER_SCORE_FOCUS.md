# Refocused Approach: Predicting Application Review Scores

## The Core Truth

**Application Review Score (0-25)** from trusted reviewers is our ground truth. Everything should be optimized to predict this score with high fidelity.

## Key Insight from Data

Looking at the Application Review Score distribution:
- Mean: 16.07
- Std: 7.30
- Range: 0-25

Tier mapping:
- 0-14: Very Unlikely (Tier 1)
- 15-18: Potential Review (Tier 2) 
- 19-22: Probable Interview (Tier 3)
- 23-25: Very Likely Interview (Tier 4)

The challenge: Many applicants cluster around 15-18 (the difficult middle zone).

## Revised LLM Output Structure

The LLM should output numeric scores that correlate with what makes reviewers give high vs low scores:

```json
{
  // Factors that strongly influence reviewer scores
  "llm_narrative_coherence": 8.0,        // How well the story flows (0-10)
  "llm_motivation_authenticity": 7.5,     // Genuine vs generic (0-10)
  "llm_reflection_depth": 8.5,            // Surface vs deep insight (0-10)
  "llm_growth_demonstrated": 9.0,         // Personal development shown (0-10)
  "llm_unique_perspective": 6.0,          // What makes them stand out (0-10)
  "llm_clinical_insight": 7.0,            // Understanding of medicine (0-10)
  "llm_service_genuineness": 8.0,         // Authentic commitment (0-10)
  "llm_leadership_impact": 6.5,           // Actual impact created (0-10)
  "llm_communication_quality": 8.0,       // Writing effectiveness (0-10)
  "llm_maturity_score": 7.5,              // Professional maturity (0-10)
  
  // Red/Green flags that reviewers notice
  "llm_red_flag_severity": 0,             // 0-10: How concerning
  "llm_green_flag_strength": 7,           // 0-10: How impressive
  
  // Overall assessment
  "llm_essay_overall_score": 75,          // 0-100: Overall quality
  
  // Direct prediction attempt
  "llm_predicted_review_score": 17.5      // 0-25: What reviewers might give
}
```

## Integration Formula

Instead of arbitrary weights, we train the model to learn what combination best predicts reviewer scores:

```python
# All features go into regression model
features = [
    # Structured (from Excel)
    gpa_total, mcat_total, research_hours, clinical_hours,
    service_rating, gpa_trend, disadvantaged_ind,
    
    # LLM scores (from essays)
    llm_narrative_coherence, llm_motivation_authenticity,
    llm_reflection_depth, llm_clinical_insight,
    
    # Interaction features
    gpa × llm_academic_readiness,  # Academic package
    clinical_hours × llm_clinical_insight,  # Clinical preparedness
    service_rating × llm_service_genuineness  # Service authenticity
]

# Model learns optimal weights through training
predicted_review_score = model.predict(features)  # 0-25
```

## What Changes

### 1. Target Variable
- **Before**: Trying to predict tiers (1-4)
- **Now**: Predict exact Application Review Score (0-25)
- **Why**: More granular, directly matches reviewer output

### 2. Model Type
- **Before**: Classification (4 classes)
- **Now**: Regression (continuous 0-25)
- **Why**: Captures nuance in middle zones (15-18)

### 3. LLM Prompt Focus
- **Before**: General quality assessment
- **Now**: "What would make reviewers score this high/low?"
- **Why**: Directly optimizes for reviewer patterns

### 4. Feature Engineering
- **Before**: Generic interactions
- **Now**: Features that correlate with reviewer scores
- **Why**: Data-driven approach

## Implementation Steps

1. **Analyze Historical Data**
   ```python
   # Find what correlates with Application Review Score
   correlations = df.corr()['Application Review Score'].sort_values()
   ```

2. **Design LLM Scoring**
   - Focus on aspects reviewers value
   - Score dimensions that predict review scores

3. **Train Regression Model**
   ```python
   # Predict continuous score
   model = XGBRegressor()
   model.fit(X_train, y_train['Application Review Score'])
   ```

4. **Optimize for Tier 2/3 Distinction**
   - Score threshold 18.5 is critical
   - Focus feature engineering on this boundary

## Success Metrics

1. **Mean Absolute Error**: Target < 2.0 points (8% of scale)
2. **Tier Accuracy**: >85% correct tier placement
3. **Adjacent Tier Accuracy**: >95% within one tier
4. **Correlation**: >0.85 with actual reviewer scores
5. **Tier 2/3 Precision**: >80% correct on boundary cases

## Key Advantages

1. **Direct Optimization**: Predicting exactly what we care about
2. **Granular Predictions**: 17.2 vs 17.8 matters for borderline cases
3. **Interpretable**: Can explain why score is predicted
4. **Calibrated**: Matches reviewer scoring patterns

The LLM becomes a tool to extract numeric features that predict reviewer behavior, not an independent evaluator. This should significantly improve prediction accuracy, especially in the critical 15-18 score range.