# Azure OpenAI Setup Guide for Medical Admissions Essay Scoring

## Overview

This guide explains how to set up and optimize Azure OpenAI for consistent, reliable essay scoring in medical admissions. The system uses GPT-4 with carefully tuned parameters to evaluate unstructured text data.

## Why These Settings?

### Temperature: 0.15
- **Why this value**: Low temperature ensures consistent scoring across essays
- **Effect**: Reduces randomness while allowing slight variation for edge cases
- **Medical context**: Critical for fair evaluation - same essay should get similar scores

### Top-p: 0.9
- **Why this value**: Focused but not overly restrictive
- **Effect**: Uses top 90% probability tokens, avoiding completely random outputs
- **Medical context**: Balances consistency with ability to recognize unique qualities

### No Vector Database Needed
- **Why**: Each essay is evaluated independently with a comprehensive rubric
- **Simpler**: No need for similarity searches or embeddings storage
- **Cost-effective**: Only pay for API calls, not vector storage
- **Sufficient**: The detailed prompt provides all context needed

## Azure Setup Steps

### 1. Create Azure OpenAI Resource

```bash
# Using Azure CLI
az cognitiveservices account create \
  --name "medical-admissions-ai" \
  --resource-group "your-resource-group" \
  --kind "OpenAI" \
  --sku "S0" \
  --location "eastus" \
  --yes
```

### 2. Deploy GPT-4 Model

1. Go to Azure OpenAI Studio
2. Click "Deployments"
3. Create new deployment:
   - Model: `gpt-4` (or `gpt-4-32k` for longer essays)
   - Deployment name: `medical-admissions-gpt4`
   - Tokens per minute: 40K (adjust based on volume)

### 3. Get Credentials

```python
# Find these in Azure Portal
AZURE_OPENAI_ENDPOINT = "https://medical-admissions-ai.openai.azure.com/"
AZURE_OPENAI_KEY = "your-key-here"
DEPLOYMENT_NAME = "medical-admissions-gpt4"
```

## Prompt Engineering Details

### System Prompt Structure

The system prompt includes:

1. **Role Definition**: Expert medical admissions evaluator with 20+ years experience
2. **Evaluation Framework**: 8 specific dimensions with clear criteria
3. **Scoring Guidelines**: 1-10 scale with defined ranges
4. **Output Format**: Structured JSON for easy integration

### Key Evaluation Dimensions

1. **Motivation for Medicine** (1-10)
   - Clear understanding of profession
   - Genuine reasons
   - Long-term commitment evidence

2. **Clinical Exposure** (1-10)
   - Depth of experiences
   - Patient care understanding
   - Reflection quality

3. **Service Orientation** (1-10)
   - Altruism evidence
   - Community involvement
   - Sustained activities

4. **Resilience & Growth** (1-10)
   - Overcoming challenges
   - Learning from failures
   - Personal maturity

5. **Academic Preparedness** (1-10)
   - Understanding of rigor
   - Scientific thinking
   - Intellectual curiosity

6. **Interpersonal Skills** (1-10)
   - Teamwork examples
   - Communication abilities
   - Cultural competence

7. **Leadership Potential** (1-10)
   - Initiative taking
   - Influence on others
   - Healthcare vision

8. **Ethical Reasoning** (1-10)
   - Medical ethics understanding
   - Moral reasoning
   - Professional behavior

### Additional Outputs

- **Red Flags**: Array of concerns
- **Green Flags**: Exceptional qualities
- **Overall Score**: 1-100 holistic assessment
- **Recommendation**: Strongly Recommend / Recommend / Neutral / Not Recommend
- **Summary**: 2-3 sentence evaluation

## Storage Strategy

### Local Caching (Recommended)
```python
# Using shelve for simple key-value storage
cache_path = "cache/essay_scores.db"
```

**Benefits**:
- No additional infrastructure
- Fast lookups
- Automatic deduplication

### Redis Caching (For Scale)
```python
# For high-volume processing
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

**Benefits**:
- Distributed caching
- TTL support
- Concurrent access

### Why Not Vector Database?

1. **Independent Evaluation**: Each essay judged on its own merits
2. **No Similarity Search Needed**: Not comparing essays to each other
3. **Structured Output**: JSON scores integrate directly with ML model
4. **Cost**: Vector databases add complexity without benefit for this use case

## Performance Optimization

### Batch Processing
```python
# Process 20 essays concurrently
batch_size = 20
```

### Rate Limiting
- Azure OpenAI: 40K tokens/minute (default)
- ~15-20 essays per minute with current prompt
- Automatic retry with exponential backoff

### Caching Strategy
- Cache for 7 days (configurable)
- Key: MD5 hash of essay + context
- Reduces duplicate API calls by ~30%

## Cost Estimation

### Per Essay
- Input: ~1000 tokens (essay + prompt)
- Output: ~200 tokens (structured response)
- Cost: ~$0.04 per essay (GPT-4)

### Volume Pricing
- 1,000 applicants: ~$40
- 5,000 applicants: ~$200
- 10,000 applicants: ~$400

### Cost Optimization
1. Use caching to avoid re-scoring
2. Batch process during off-peak
3. Consider GPT-3.5-Turbo for initial screening

## Integration Example

```python
# Initialize Azure scorer
azure_config = {
    'azure_endpoint': 'https://medical-admissions-ai.openai.azure.com/',
    'api_key': 'your-key',
    'deployment_name': 'medical-admissions-gpt4',
    'api_version': '2024-02-15-preview'
}

scorer = AzureEssayScorer(**azure_config)

# Score an essay
result = await scorer.score_essay_async(
    essay_text="My journey to medicine began...",
    additional_context={
        'age': 24,
        'total_hours': 2000,
        'research_hours': 500
    }
)

# Result includes all dimensions + recommendations
print(f"Overall Score: {result['overall_score']}/100")
print(f"Recommendation: {result['recommendation']}")
```

## Monitoring & Quality Control

### Metrics to Track
1. **Consistency**: Score variance for similar essays
2. **Distribution**: Ensure reasonable score distribution
3. **Red Flags**: Monitor frequency and types
4. **API Performance**: Response times and errors

### Quality Checks
```python
# Periodic validation
test_essays = load_test_set()
scores = await scorer.batch_score_essays(test_essays)

# Check score distribution
assert scores['overall_score'].mean() between 40-60
assert scores['overall_score'].std() > 10  # Sufficient differentiation
```

## Security Considerations

1. **API Key Storage**: Use Azure Key Vault
2. **Data Privacy**: Don't log full essays
3. **Access Control**: Limit API access by IP
4. **Audit Logging**: Track all scoring requests

## Troubleshooting

### Common Issues

1. **Inconsistent Scores**
   - Check temperature isn't too high
   - Ensure prompt hasn't changed
   - Verify model deployment

2. **Timeout Errors**
   - Reduce batch size
   - Increase timeout settings
   - Check essay length

3. **JSON Parse Errors**
   - Verify response_format setting
   - Check for prompt injection
   - Add validation layer

## Summary

This Azure OpenAI setup provides:
- ✅ Consistent, fair essay evaluation
- ✅ Structured output for ML integration  
- ✅ No vector database complexity
- ✅ Cost-effective scaling
- ✅ Easy integration with existing pipeline

The low temperature (0.15) and focused top-p (0.9) ensure reliable scoring while the comprehensive prompt eliminates the need for vector similarity searches.