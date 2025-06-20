# Medical Admissions Model: Complete Architecture Summary

## 🎯 Core Objective
**Predict Application Review Score (0-25)** with high fidelity to identify interview candidates.

## 📊 Key Insights from Data

1. **Interview Threshold: Score ≥ 19** (NOT 18)
   - Based on actual data analysis
   - ~50% of applicants score ≥19

2. **No Interview Column Exists**
   - Must infer from Application Review Score
   - Scores 19-25 → Interview likely
   - Scores 0-18 → No interview

3. **Score Distribution**
   - Mean: ~16
   - Median: 17-19
   - Critical boundary: 18-19 (Tier 2/3 distinction)

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT DATA SOURCES                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STRUCTURED DATA (Excel)          UNSTRUCTURED DATA         │
│  ├── 1. Applicants.xlsx          ├── 9. Personal Statement │
│  ├── 2. Language.xlsx            ├── 10. Secondary Essays  │
│  ├── 3. Parents.xlsx             └── 6. Experience Descrip. │
│  ├── 4. Siblings.xlsx                       │               │
│  ├── 5. Academic Records.xlsx               │               │
│  ├── 8. Schools.xlsx                        ▼               │
│  └── 12. GPA Trend.xlsx            Azure OpenAI GPT-4      │
│           │                         (Essay Evaluation)       │
│           ▼                                 │               │
│    Feature Engineering                      ▼               │
│    ├── One-hot encoding            LLM Numeric Scores      │
│    ├── Ratio calculations          ├── Motivation: 8.5     │
│    ├── Diversity scores            ├── Clinical: 7.0       │
│    └── Trend analysis              ├── Leadership: 6.5     │
│           │                        └── Overall: 75/100     │
│           └────────────┬────────────────────┘               │
│                        ▼                                    │
│              COMBINED FEATURE VECTOR                        │
│              [125 structured + 13 LLM + 12 interactions]   │
│                        │                                    │
│                        ▼                                    │
│                 XGBoost Regressor                           │
│              (Trained on 2022-2023)                         │
│                        │                                    │
│                        ▼                                    │
│              Predicted Review Score                         │
│                    (0-25)                                   │
│                        │                                    │
│                        ▼                                    │
│              Decision Thresholds                            │
│              ├── <19: No Interview                         │
│              └── ≥19: Interview                            │
└─────────────────────────────────────────────────────────────┘
```

## 🧮 Feature Categories

### 1. **Structured Features** (~125 after encoding)
- **Academic**: GPA, MCAT, course diversity, trends
- **Experience**: Research, clinical, volunteer hours
- **Diversity**: First-gen, disadvantaged, rural, languages
- **Financial**: Income level, Pell grant, assistance

### 2. **LLM Features** (13 numeric scores)
```python
llm_motivation_score: 0-10       # How genuine is their calling?
llm_clinical_insight: 0-10       # Do they understand medicine?
llm_reflection_depth: 0-10       # Surface vs deep thinking
llm_leadership_impact: 0-10      # Real impact vs titles
llm_red_flag_severity: 0-10      # Concerning issues
llm_essay_overall: 0-100         # General quality
```

### 3. **Interaction Features** (~12)
```python
academic_package = GPA × LLM_academic_readiness
clinical_preparedness = Clinical_hours × LLM_clinical_insight
service_authenticity = Service_rating × LLM_service_genuine
```

## 🎯 Target Variable Options

1. **Primary**: Continuous Application Review Score (0-25)
2. **Binary**: Interview (≥19) vs No Interview (<19)
3. **Quartiles**: Low, Medium-Low, Medium-High, High
4. **Tiers**: Very Unlikely (0-14), Potential (15-18), Probable (19-22), Very Likely (23-25)

## 🤖 Model Selection

### Primary: **XGBoost Regressor**
- Best for mixed data types
- Captures complex interactions
- Handles missing values
- Built-in feature importance

### Alternatives to Test:
- Random Forest (baseline)
- Neural Network (deep interactions)
- Ensemble (combine all three)

## 📈 Expected Performance

- **Score Prediction**: MAE ~2.0 points (8% of scale)
- **Interview Decision**: 85-90% accuracy
- **Tier 2/3 Distinction**: 70-75% precision
- **Feature Importance**: 
  - Structured: ~65%
  - LLM Essays: ~25%
  - Interactions: ~10%

## 🔑 Key Success Factors

1. **Rich Feature Set**: 150+ features capturing holistic view
2. **LLM Integration**: Essay quality quantified numerically
3. **Interaction Terms**: Multiplicative effects (quality × quantity)
4. **Proper Validation**: Train on 2022-2023, test on 2024
5. **Focus on Boundary**: Special attention to scores 17-20

## 💡 Critical Implementation Notes

- **One-hot encode** categoricals (Gender, Major, School)
- **Scale** continuous features (hours, scores)
- **Handle missing** financial aid data (~11% missing)
- **Weight samples** near decision boundary
- **Monitor** for bias in predictions

This architecture predicts what reviewers actually scored, not an independent assessment, ensuring high fidelity to the ground truth.