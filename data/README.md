# Data Directory

This directory contains ONLY synthetic/fake data for testing and demonstration purposes.

## ⚠️ CRITICAL PRIVACY NOTICE ⚠️

**NEVER store real applicant data in this repository**

- No real AMCAS IDs
- No actual essays or personal statements  
- No genuine application metrics
- No personally identifiable information (PII)
- No FERPA-protected educational records
- No HIPAA-protected health information

## Directory Structure

```
data/
├── sample/               # Synthetic sample data ONLY
│   ├── sample_applications.csv
│   └── README.md
└── README.md            # This file
```

## Synthetic Data

All data in this directory is:
- Artificially generated
- Uses fake IDs (FAKE001, FAKE002, etc.)
- Contains generic essay descriptions
- Has randomized metrics within realistic ranges

## For Testing

Use the synthetic data for:
- Development and debugging
- User interface testing
- Training and demonstrations
- Documentation examples

## Generate More Test Data

```bash
python scripts/generate_synthetic_data.py --count 100
```

## Production Data Handling

Real application data should:
- NEVER be committed to Git
- Be stored in secure, HIPAA-compliant systems
- Be accessed only through secure APIs
- Follow institutional data governance policies
- Be encrypted at rest and in transit

## Compliance

This approach ensures compliance with:
- FERPA (Family Educational Rights and Privacy Act)
- HIPAA (Health Insurance Portability and Accountability Act)
- Institutional Review Board (IRB) requirements
- Ethical AI development practices

## Questions?

Contact the data governance team before handling any real applicant data.