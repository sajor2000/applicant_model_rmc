# Medical Admissions AI - Quick Start Guide

## Overview

This system combines:
1. **Machine Learning models** trained on 2022-2023 structured applicant data
2. **Azure OpenAI GPT-4** for essay/unstructured text evaluation  
3. **Integrated scoring** that weighs both components for final recommendations

**Training Data**: 2022 + 2023 combined  
**Test Data**: 2024 (held out, never seen during training)

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set Azure credentials (required for essay scoring)
export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
export AZURE_OPENAI_KEY='your-api-key'
export AZURE_OPENAI_DEPLOYMENT='gpt-4'  # or your deployment name
```

## Step 1: Train the Model (One-time)

Train on 2022-2023 data, test on 2024:

```bash
python train_test_2024.py
```

This will:
- Load and combine 2022-2023 training data
- Load 2024 as separate test set
- Train multiple models (Random Forest, XGBoost, etc.)
- Evaluate all models on 2024 data
- Save the best model to `models/best_model_2022_2023_train_2024_test.pkl`

Expected output:
```
MEDICAL ADMISSIONS MODEL TRAINING PIPELINE
Training Data: 2022 + 2023 Combined
Test Data: 2024 (Holdout Set)
================================================================================

[STEP 1] Loading Training Data (2022 + 2023)
✓ Loaded 2022 data: 437 applicants
✓ Loaded 2023 data: 523 applicants
✓ Combined training data: 960 total applicants

[STEP 2] Loading Test Data (2024 Holdout)
✓ Loaded 2024 test data: 412 applicants

Best Model: RandomForest
Test Performance (2024): 0.823 weighted F1
```

## Step 2: Test Azure Setup

Verify Azure OpenAI is configured correctly:

```bash
python test_azure_setup.py
```

This will:
- Check Azure credentials
- Test essay scoring with sample essays
- Verify scoring consistency
- Show performance metrics

Expected output shows different scores for strong/average/weak candidates.

## Step 3: Run Integrated Evaluation

Process applicants with both structured data and essays:

```bash
python integrated_azure_pipeline.py
```

This will:
- Load applicant data and essays
- Score structured features with ML model
- Score essays with Azure OpenAI
- Calculate integrated scores
- Export comprehensive results

Output files in `integrated_results/`:
- `integrated_evaluation_TIMESTAMP.xlsx` (multi-sheet Excel)
- `integrated_evaluation_TIMESTAMP.csv` (simple format)

## Step 4: Bulk Processing (Optional)

For high-volume processing without essay analysis:

```bash
python enhanced_bulk_processor.py
```

Features:
- Batch processing with progress bars
- Parallel execution
- Multiple export formats
- Tier-based categorization

## Understanding the Results

### Excel Sheets

1. **All_Applicants**: Complete results sorted by score
2. **Interview_Candidates**: Tier 3-4 (Probable/Very Likely Interview)
3. **High_Confidence**: Predictions with >80% confidence
4. **Essay_Analysis**: Detailed essay scoring breakdown

### Key Columns

- `integrated_tier_name`: Final recommendation tier
- `integrated_confidence`: Model confidence (0-1)
- `structured_prediction`: ML model tier (0-3)
- `essay_scores`: Detailed essay evaluation
- `recommendation`: Essay-based recommendation
- `key_factors`: Important decision factors

### Tier Definitions

1. **Very Unlikely** (Tier 1): Not recommended for interview
2. **Potential Review** (Tier 2): Borderline, needs committee review
3. **Probable Interview** (Tier 3): Strong candidate for interview
4. **Very Likely Interview** (Tier 4): Top candidates

## Azure OpenAI Configuration

### Why These Settings?

- **Temperature: 0.15** - Ensures consistent scoring
- **Top-p: 0.9** - Focused but not restrictive
- **No Vector DB** - Each essay evaluated independently with rubric

### Essay Evaluation Dimensions

1. Motivation for Medicine (1-10)
2. Clinical Understanding (1-10)
3. Service Commitment (1-10)
4. Resilience & Growth (1-10)
5. Academic Preparedness (1-10)
6. Interpersonal Skills (1-10)
7. Leadership Potential (1-10)
8. Ethical Reasoning (1-10)

Plus: Red flags, green flags, overall score (1-100), and recommendation.

## Cost Estimates

- **Per applicant**: ~$0.04 (with essay scoring)
- **1,000 applicants**: ~$40
- **5,000 applicants**: ~$200

Without essay scoring (structured data only): $0

## Troubleshooting

### "Model file not found"
Run `python train_test_2024.py` first to train the model.

### "Azure credentials not found"
Set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT='...'
export AZURE_OPENAI_KEY='...'
```

### Low essay scores for all applicants
Check Azure deployment is using GPT-4 (not GPT-3.5).

### Slow processing
- Reduce batch size in scripts
- Check Azure rate limits
- Enable caching (default: on)

## Advanced Usage

### Custom Weights
Adjust essay vs structured data weights in `integrated_azure_pipeline.py`:
```python
system = IntegratedAzureAdmissionsSystem(
    essay_weight=0.35,      # 35% essay
    structured_weight=0.65  # 65% structured data
)
```

### Add More Years
Modify `train_test_2024.py` to include additional training years:
```python
# In load_and_combine_training_data()
years_to_load = [2021, 2022, 2023]  # Add more years
```

## Support

For issues or questions:
1. Check error messages in console
2. Review log files in respective output directories
3. Verify data format matches expected structure
4. Ensure all Excel files have correct column names