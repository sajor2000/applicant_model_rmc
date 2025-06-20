# Sample Data

This directory contains **synthetic/fake data** for demonstration and testing purposes only.

## Important Privacy Notice

⚠️ **This is synthetic data only**
- All applicant IDs are fake (FAKE001, FAKE002, etc.)
- All essays are artificially generated examples
- All metrics are randomly generated within realistic ranges
- **No real applicant data is included**

## File Contents

### sample_applications.csv

Contains 10 synthetic applications with:
- Fake AMCAS IDs
- Randomly generated metrics within typical ranges
- Generic essay summaries (not real essays)
- Diverse demographic representations

## Usage

This sample data can be used for:
- Testing the application processing pipeline
- Demonstrating the web interface
- Training new users
- Development and debugging

## Data Format

| Column | Type | Description |
|--------|------|-------------|
| amcas_id | String | Fake identifier (FAKE###) |
| service_rating_numerical | Integer (1-4) | Synthetic service rating |
| healthcare_total_hours | Integer | Random clinical hours |
| exp_hour_research | Integer | Random research hours |
| exp_hour_volunteer_med | Integer | Random medical volunteering |
| exp_hour_volunteer_non_med | Integer | Random non-medical volunteering |
| age | Integer | Age within typical range (22-30) |
| gender | String | Male/Female/Other |
| citizenship | String | US_Citizen/Permanent_Resident/International |
| first_generation_ind | Binary | 0 or 1 |
| essay_text | String | Generic essay description (NOT real essays) |

## Generating More Sample Data

To generate additional synthetic data for testing:

```python
python scripts/generate_synthetic_data.py --count 100
```

## Privacy Compliance

This synthetic data ensures:
- No PII (Personally Identifiable Information)
- No real application content
- FERPA compliance
- HIPAA compliance
- Ethical testing practices

## Do NOT:
- Replace with real applicant data
- Use actual AMCAS IDs
- Include real essays or personal statements
- Share any real application information