# Rush Medical College AI Admissions System
## Comprehensive Feature Analysis & Model Results - 2024

---

## Model Architecture Overview

### AI Models Used:
1. **Essay Analysis**: Azure OpenAI GPT-4o (GPT-4-omnibus, latest multimodal model)
   - Temperature: 0.15 (for consistent scoring)
   - Top-p: 0.9
   - Used for extracting 13 essay-based features

2. **Ranking Model**: XGBoost Cascade Classifier
   - 3-stage binary cascade
   - 5-model ensemble per stage
   - Isotonic probability calibration

### Training Data: 
- 2022 cohort: 473 applicants
- 2023 cohort: 365 applicants
- Combined: 838 applicants

### Test Data:
- 2024 cohort: 613 applicants (holdout set)

---

## Complete Feature Dictionary

### Table 1: Structured Application Features (Original Data)

| Feature Name | Definition | Data Type | Example Values | Missing Rate |
|--------------|------------|-----------|----------------|--------------|
| **service_rating_numerical** | Faculty-assigned service quality score | Integer (1-4) | 1=Poor, 2=Average, 3=Good, 4=Exceptional | 0% |
| **healthcare_total_hours** | Total hours of healthcare-related experience | Integer | 0-10,000+ hours | 3.2% |
| **exp_hour_clinical** | Hours of direct clinical experience | Integer | 0-5,000 hours | 8.1% |
| **exp_hour_research** | Hours devoted to research activities | Integer | 0-8,000 hours | 5.4% |
| **exp_hour_volunteer_med** | Medical volunteering hours | Integer | 0-2,000 hours | 4.7% |
| **exp_hour_volunteer_non_med** | Non-medical volunteering hours | Integer | 0-3,000 hours | 6.2% |
| **exp_hour_shadowing** | Physician shadowing hours | Integer | 0-500 hours | 12.3% |
| **exp_hour_employment** | Paid employment hours (any type) | Integer | 0-15,000 hours | 9.8% |
| **exp_hour_leadership** | Leadership activity hours | Integer | 0-2,000 hours | 15.6% |
| **exp_hour_teaching** | Teaching/tutoring hours | Integer | 0-1,500 hours | 18.4% |
| **age** | Applicant age at time of application | Integer | 20-45 years | 0% |
| **gender** | Self-reported gender | Categorical | Male/Female/Other/Unknown | 0.2% |
| **citizenship** | Citizenship status | Categorical | US_Citizen/Permanent_Resident/International | 0% |
| **first_generation_ind** | First in family to attend college | Binary | 0=No, 1=Yes | 2.1% |
| **pell_grant_ind** | Received Pell Grant (low income) | Binary | 0=No, 1=Yes | 3.5% |
| **disadvantaged_ind** | Self-identified disadvantaged background | Binary | 0=No, 1=Yes | 1.8% |
| **service_rating_categorical** | Service rating as text | Categorical | Poor/Average/Good/Excellent/Outstanding/Exceptional | 0% |

### Table 2: GPT-4o Essay Analysis Features

| Feature Name | Definition | Scale | What GPT-4o Evaluates | Prompt Focus |
|--------------|------------|-------|----------------------|--------------|
| **llm_overall_essay_score** | Composite essay quality | 0-100 | Writing quality, coherence, impact, authenticity | "Rate the overall quality and persuasiveness" |
| **llm_motivation_authenticity** | Genuineness of medical motivation | 0-100 | Authentic vs formulaic reasons for medicine | "How genuine and personal are the motivations" |
| **llm_clinical_insight** | Understanding of clinical practice | 0-100 | Realistic view of medicine, patient care understanding | "Depth of clinical understanding demonstrated" |
| **llm_leadership_impact** | Leadership effectiveness shown | 0-100 | Concrete examples, measurable impact, initiative | "Evidence of leadership with tangible outcomes" |
| **llm_service_genuineness** | Authenticity of service orientation | 0-100 | Selfless service vs resume building | "Genuine commitment to serving others" |
| **llm_intellectual_curiosity** | Love of learning demonstrated | 0-100 | Research interest, questioning mind, growth mindset | "Evidence of intellectual engagement" |
| **llm_maturity_score** | Emotional/professional maturity | 0-100 | Self-reflection, growth from challenges, judgment | "Emotional maturity and self-awareness" |
| **llm_communication_score** | Writing clarity and persuasiveness | 0-100 | Clear expression, organization, compelling narrative | "Effectiveness of written communication" |
| **llm_diversity_contribution** | Unique perspective offered | 0-100 | Background, experiences, viewpoints that add value | "Unique contributions to medical school diversity" |
| **llm_resilience_score** | Overcoming challenges | 0-100 | Response to adversity, persistence, growth | "Evidence of resilience and perseverance" |
| **llm_ethical_reasoning** | Ethical thinking demonstrated | 0-100 | Moral reasoning, integrity, professional values | "Understanding of medical ethics and integrity" |
| **llm_red_flag_count** | Concerning content flags | Count | Professionalism lapses, unrealistic views, concerning behavior | "Number of concerning elements" |
| **llm_green_flag_count** | Exceptional content flags | Count | Outstanding achievements, exceptional insight, unique strengths | "Number of exceptional elements" |

### Table 3: Engineered Features (Created During Model Development)

| Feature Name | Definition | Calculation | Purpose |
|--------------|------------|-------------|---------|
| **essay_service_alignment** | Coherence between essays and service rating | 1 - abs(essay_norm - service_norm) | Detect authentic vs manufactured narratives |
| **service_essay_product** | Interaction: service × essay quality | service_rating × llm_overall_score / 25 | Capture synergy between demonstrated and written |
| **profile_coherence** | Overall application consistency | Mean correlation of essay and activity features | Identify well-integrated applications |
| **experience_consistency** | Balance across experiences | 1 / (1 + std(experience_hours) / mean) | Detect well-rounded vs narrow applicants |
| **experience_diversity** | Number of different activities | Count of activities with >50 hours | Breadth of engagement |
| **flag_balance** | Net positive indicators | green_flags - red_flags | Overall positive/negative signal |
| **flag_ratio** | Relative flag positivity | green_flags / (red_flags + 1) | Proportional flag assessment |
| **service_clinical_log** | Service × clinical interaction | service_rating × log(clinical_hours + 1) | High service + high clinical exposure |
| **clinical_research_ratio** | Clinical vs research balance | clinical_hours / (research_hours + 1) | Identify clinical vs research focused |
| **academic_potential_score** | Composite academic indicator | Mean of normalized(research + curiosity + maturity) | Research and intellectual readiness |
| **clinical_readiness_score** | Composite clinical indicator | Mean of normalized(clinical_hours + clinical_insight + shadowing) | Clinical preparation level |
| **service_squared** | Non-linear service impact | service_rating² | Capture exceptional service impact |
| **age_first_gen** | Age × first generation | age × first_generation_ind | Interaction of non-traditional markers |
| **extreme_value_count** | Number of outlier features | Count of values in top/bottom 5% | Identify unusual profiles |

---

## Feature Importance Results

### Top 20 Most Important Features (After Model Training)

| Rank | Feature | Importance | Category | Interpretation |
|------|---------|------------|----------|----------------|
| 1 | **service_rating_numerical** | 18.2% | Structured | Faculty service evaluation is #1 predictor |
| 2 | **service_essay_product** | 7.9% | Engineered | Service × essay synergy highly predictive |
| 3 | **llm_service_genuineness** | 7.1% | GPT-4o | Authentic service commitment (essay) |
| 4 | **healthcare_total_hours** | 6.8% | Structured | Total clinical exposure |
| 5 | **profile_coherence** | 5.9% | Engineered | Application consistency matters |
| 6 | **llm_overall_essay_score** | 5.2% | GPT-4o | Overall essay quality |
| 7 | **experience_consistency** | 4.7% | Engineered | Balanced experiences valued |
| 8 | **llm_maturity_score** | 4.3% | GPT-4o | Emotional/professional maturity |
| 9 | **flag_balance** | 3.9% | Engineered | Net positive indicators |
| 10 | **clinical_readiness_score** | 3.6% | Engineered | Composite clinical preparation |
| 11 | **service_clinical_log** | 3.2% | Engineered | Service × clinical hours interaction |
| 12 | **age** | 2.9% | Structured | Life experience factor |
| 13 | **llm_clinical_insight** | 2.7% | GPT-4o | Understanding of medicine |
| 14 | **essay_consistency** | 2.4% | Engineered | Essay score variance |
| 15 | **academic_potential_score** | 2.1% | Engineered | Research/intellectual composite |
| 16 | **exp_hour_research** | 1.9% | Structured | Research hours (less important than expected) |
| 17 | **llm_leadership_impact** | 1.7% | GPT-4o | Leadership effectiveness |
| 18 | **first_generation_ind** | 1.5% | Structured | First-gen college status |
| 19 | **llm_communication_score** | 1.3% | GPT-4o | Writing effectiveness |
| 20 | **exp_hour_volunteer_med** | 1.1% | Structured | Medical volunteering |

---

## Model Training Process (2022-2023 Data)

### Step 1: Data Preparation
- Combined 2022 (n=473) and 2023 (n=365) cohorts
- Standardized column names (critical fix for service_rating_numerical)
- Imputed missing values using median imputation
- Scaled all features to mean=0, std=1

### Step 2: Essay Processing with GPT-4o
- Processed 838 essays through Azure OpenAI GPT-4o
- Extracted 13 essay features per applicant
- Total cost: $33.52 for training set
- Processing time: ~25 minutes

### Step 3: Feature Engineering
- Created 15 interaction and composite features
- Identified extreme values (top/bottom 5%)
- Built coherence and consistency metrics

### Step 4: Model Architecture
```
Cascade Stage 1: Reject (≤9) vs Non-Reject (>9)
- Training samples: 114 reject, 724 non-reject
- Features used: All 73 features
- Model: XGBoost ensemble (5 models)
- Cross-validation AUC: 0.945

Cascade Stage 2: Waitlist (10-15) vs Higher (>15)  
- Training samples: 282 waitlist, 442 higher
- Features used: All 73 features
- Model: XGBoost ensemble (5 models)
- Cross-validation AUC: 0.843

Cascade Stage 3: Interview (16-22) vs Accept (≥23)
- Training samples: 370 interview, 72 accept
- Features used: All 73 features
- Model: XGBoost ensemble (5 models)
- Cross-validation AUC: 0.894
```

---

## Testing on 2024 Holdout Data

### Test Process:
1. Loaded 613 new 2024 applicants (never seen during training)
2. Processed essays through GPT-4o (same prompts/parameters)
3. Applied identical feature engineering
4. Made predictions using trained cascade model

### Results Summary:

| Metric | Result | Details |
|--------|--------|---------|
| **Exact Quartile Match** | 80.8% | 495/613 correct |
| **Adjacent Accuracy** | 99.0% | 607/613 within 1 quartile |
| **Major Errors** | 1.0% | 6/613 off by >1 quartile |

### Confusion Matrix:
```
              Predicted
              Q4    Q3    Q2    Q1
Actual  Q4    97    24     6     0   (Reject/Lowest)
        Q3    12    75    34     0   (Waitlist)
        Q2     0    11   260    22   (Interview)  
        Q1     0     0     9    63   (Accept/Highest)
```

### Performance by Score Range:
- Scores 0-9 (Reject): 76.4% correctly identified
- Scores 10-15 (Waitlist): 62.0% correctly identified
- Scores 16-22 (Interview): 88.7% correctly identified
- Scores 23-25 (Accept): 87.5% correctly identified

---

## Key Insights from Feature Analysis

### 1. Service Dominates
- Service rating (18.2%) + service interactions (11.1%) = 29.3% of model
- Service is 4x more predictive than research hours
- Authentic service (from essays) matters as much as rated service

### 2. Essays Add Crucial Information
- GPT-4o features comprise 35% of decision weight
- Essay-structure alignment highly predictive
- Maturity and genuineness cannot be captured from activities alone

### 3. Quality Over Quantity
- Experience consistency (4.7%) > Total volunteer hours (1.1%)
- Clinical insight (2.7%) > Shadowing hours (0.8%)
- Profile coherence (5.9%) > Any single activity

### 4. Interactions Matter Most
- 6 of top 20 features are engineered interactions
- Service × Essay product is #2 overall predictor
- Coherent narratives that match activities predict success

---

## Fairness Validation

All features were tested for bias across:
- Gender (no significant differences, p=0.976)
- First-generation status (equitable representation)
- Age (minimal correlation with outcomes)
- Socioeconomic status (no Pell Grant penalty)

---

*Report Date: June 19, 2025*  
*Model: GPT-4o (Azure OpenAI) + XGBoost Cascade*  
*Training: 2022-2023 applicants (n=838)*  
*Testing: 2024 applicants (n=613)*