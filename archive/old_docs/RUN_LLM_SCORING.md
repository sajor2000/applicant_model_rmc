# Running the LLM Scoring Pipeline

## Prerequisites

1. **Set Azure OpenAI Environment Variables**:
```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

2. **Install Required Packages**:
```bash
pip install openai pandas numpy tqdm tenacity openpyxl
```

3. **Ensure Data Files Exist**:
- `data/2022 Applicants Reviewed by Trusted Reviewers/`
- `data/2023 Applicants Reviewed by Trusted Reviewers/`

## Step-by-Step Execution

### Option 1: Run Complete Pipeline
```bash
python step3_batch_processing.py
```

This will:
1. Connect to Azure OpenAI
2. Extract all essays from 2022-2023 data
3. Process each applicant through the LLM
4. Save results to `llm_scores_2022_2023.csv`

### Option 2: Run Each Step Individually

#### Step 1: Test Azure Connection
```python
from step1_azure_connection import AzureOpenAIConnection

# Test connection
azure = AzureOpenAIConnection()
if azure.test_connection():
    print("✅ Connection successful!")
```

#### Step 2: Extract Essays
```python
from step2_extract_unstructured_data import UnstructuredDataExtractor

# Extract data
extractor = UnstructuredDataExtractor("data")
data_2022 = extractor.extract_year_data(2022)
data_2023 = extractor.extract_year_data(2023)

# Check what we got
stats = extractor.get_summary_statistics(data_2022)
print(f"2022: {stats['total_applicants']} applicants")
```

#### Step 3: Process Through LLM
```python
from step3_batch_processing import BatchProcessor, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Process a small test batch first
processor = BatchProcessor(azure, batch_size=5)
test_data = dict(list(data_2022.items())[:5])  # First 5 applicants
test_results = processor.process_all_applicants(test_data)

# Check results
print(test_results.head())
```

## Output Format

The pipeline creates `llm_scores_2022_2023.csv` with columns:
- `AMCAS_ID`: Applicant identifier
- `year`: 2022 or 2023
- `llm_narrative_coherence`: 0-10 score
- `llm_motivation_authenticity`: 0-10 score
- `llm_reflection_depth`: 0-10 score
- `llm_growth_demonstrated`: 0-10 score
- `llm_unique_perspective`: 0-10 score
- `llm_clinical_insight`: 0-10 score
- `llm_service_genuineness`: 0-10 score
- `llm_leadership_impact`: 0-10 score
- `llm_communication_quality`: 0-10 score
- `llm_maturity_score`: 0-10 score
- `llm_red_flag_count`: Integer count
- `llm_green_flag_count`: Integer count
- `llm_overall_essay_score`: 0-100 score

## Cost and Time Estimates

- **Time**: ~2 seconds per applicant
- **Cost**: ~$0.04 per applicant (GPT-4)
- **For 1000 applicants**: ~33 minutes, ~$40

## Troubleshooting

### Connection Errors
```
❌ Azure connection failed
```
- Check environment variables are set correctly
- Verify API key is valid
- Ensure deployment name matches your Azure setup

### Missing Data
```
Warning: Personal statement file not found
```
- Check file paths match exactly (including spaces)
- Verify year folders exist
- Ensure Excel files are not open in another program

### Rate Limiting
```
Error: Rate limit exceeded
```
- Increase `delay_between_batches` in BatchProcessor
- Reduce `batch_size` to 5 or less
- Check Azure rate limits for your tier

## Integration with ML Pipeline

After running, use the scores in your ML model:

```python
# Load LLM scores
llm_scores = pd.read_csv("llm_scores_2022_2023.csv")

# Merge with structured data
structured_data = pd.read_excel("data/2022.../1. Applicants.xlsx")
combined = structured_data.merge(llm_scores, on='AMCAS_ID', how='left')

# Continue with ML pipeline
from data_transformation_pipeline import DataTransformationPipeline
pipeline = DataTransformationPipeline()
X, feature_names = pipeline.stage5_select_and_scale(combined, llm_scores)
```

## Final Prompt Being Used

The system uses this prompt to ensure numeric-only outputs:

**System Prompt**: Expert reviewer evaluating for 0-25 Application Review Score prediction

**User Prompt**: Evaluates personal statement, secondary essays, and experiences

**Output**: JSON with 13 numeric scores that predict reviewer behavior

This approach ensures consistency and ML-compatibility while capturing essay quality.