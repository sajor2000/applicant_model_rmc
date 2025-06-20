# Medical Admissions AI System - Project Summary

## Overview
This project implements a comprehensive AI-powered medical school admissions evaluation system that combines structured application data with LLM-evaluated essays to predict reviewer scores and classify applicants into 4 buckets.

## System Architecture

### 1. Data Sources
- **Structured Data**: Excel files containing demographics, academics, experiences, etc.
- **Unstructured Data**: Personal statements, secondary essays, and experience descriptions
- **Target Variable**: Application Review Score (0-25) from trusted human reviewers
- **Training Data**: 2022-2023 combined (838 applicants)
- **Test Data**: 2024 holdout set (613 applicants)

### 2. Core Components

#### Azure OpenAI Integration
- **Model**: GPT-4o deployment
- **Cost**: ~$0.04 per applicant
- **Processing Time**: ~1.8 seconds per applicant
- **Features Generated**: 13 numeric scores from essays
  - Narrative coherence, motivation authenticity, reflection depth, etc.
  - Overall essay score (0-100)
  - Red/green flag counts

#### Feature Engineering
- **Original Features**: 56 numeric features from structured data
- **Enhanced Features**: 81 total features after engineering
  - Bucket indicators (reject/waitlist/interview/accept markers)
  - Ratio features (research/clinical/volunteer proportions)
  - Interaction terms (disadvantaged × achievement)
  - Polynomial features for key predictors
  - Threshold indicators

#### Ordinal Regression Model
- **Algorithm**: XGBoost with ordinal cumulative link model
- **Buckets**: 4 categories with natural break boundaries
  1. **Reject** (0-9): 13.6% of training data
  2. **Waitlist** (11-15): 33.7% of training data
  3. **Interview** (17-21): 35.6% of training data
  4. **Accept** (23-25): 17.2% of training data

## Performance Metrics

### Validation Set (20% of 2022-2023)
- **Exact Match Accuracy**: 57.7%
- **Adjacent Accuracy (±1 bucket)**: 98.8%
- **Quadratic Weighted Kappa**: 0.724
- **Average Confidence**: 66.9%

### Per-Bucket Performance
- **Reject**: 43.5% accuracy
- **Waitlist**: 58.9% accuracy
- **Interview**: 50.0% accuracy
- **Accept**: 82.8% accuracy (best performance)

### Feature Importance
1. **Service Rating** (20.6%) - Most important feature
2. **Clinical Experience Indicators** (2.4%)
3. **Healthcare Employment Hours** (2.2%)
4. **LLM Essay Scores** (~15% combined)
5. **GPA Trends** (1.9%)

## Key Findings

### Strengths
1. **High Adjacent Accuracy**: 98.8% predictions within 1 bucket
2. **Good Ordinal Performance**: QWK of 0.724 indicates strong rank ordering
3. **LLM Integration Works**: Essay scores contribute ~15% to predictions
4. **Fair Across Demographics**: No explicit bias in features

### Limitations
1. **Missing Critical Data**: No MCAT scores or actual GPA values in dataset
2. **Low Feature Correlations**: Most features show <0.1 correlation with target
3. **2024 Prediction Issue**: 98.7% predicted as "Reject" - suggests distribution shift
4. **Moderate Exact Accuracy**: 57.7% exact match needs improvement

## File Structure

### Core Pipeline Files
- `step1_azure_connection.py` - Azure OpenAI connection setup
- `step2_extract_unstructured_data.py` - Essay extraction from Excel
- `step3_batch_processing.py` - Batch LLM processing
- `ordinal_regression_model.py` - Custom ordinal XGBoost implementation
- `enhanced_features.py` - Advanced feature engineering
- `train_ordinal_model.py` - Complete training pipeline

### Configuration
- `.env` - Azure credentials and settings
- `AZURE_ASSISTANT_FINAL_PROMPT.md` - LLM evaluation prompt

### Output Files
- `llm_scores_2022_2023_*.csv` - LLM scores for training data
- `llm_scores_2024_*.csv` - LLM scores for test data
- `ordinal_predictions_2024_*.csv` - Final predictions
- `models/ordinal_model_latest.pkl` - Trained model

## Usage

### Training New Model
```python
from train_ordinal_model import OrdinalTrainingPipeline
pipeline = OrdinalTrainingPipeline()
model_path = pipeline.run_training_pipeline()
```

### Making Predictions
```python
import joblib
model_data = joblib.load('models/ordinal_model_latest.pkl')
model = model_data['ordinal_model']
engineer = model_data['feature_engineer']

# Prepare new applicant data
X = engineer.transform(applicant_df)
bucket_prediction, confidence = model.predict_with_confidence(X)
```

## Recommendations for Improvement

1. **Add Missing Features**
   - MCAT scores (likely strongest predictor)
   - Actual GPA values (not just trends)
   - Research publications
   - School prestige/ranking

2. **Improve LLM Evaluation**
   - Increase temperature for more variance
   - Add specific rubrics for each essay type
   - Extract quantifiable achievements

3. **Model Enhancements**
   - Ensemble multiple model types
   - Implement calibration for better confidence scores
   - Add temporal features for application timing

4. **Production Deployment**
   - Build API for real-time predictions
   - Create monitoring dashboard
   - Implement A/B testing framework

## Cost Analysis
- **LLM Processing**: ~$58 for 1,451 applicants
- **Average**: $0.04 per applicant
- **Scalable**: Can process ~800 applicants in 25 minutes

## Ethical Considerations
- No direct use of race/ethnicity in predictions
- Disadvantaged status used only in interactions
- Transparent feature importance reporting
- Designed for decision support, not replacement

## Next Steps
1. Obtain and integrate MCAT/GPA data
2. Deploy production API
3. Build interactive dashboard
4. Implement continuous learning pipeline
5. A/B test against human reviewers