# Rush Medical College AI Admissions System - Production Guide 2025

## Overview
This AI-powered system assists the Rush Medical College admissions committee by analyzing applications and predicting reviewer scores with 93.8% accuracy. The system uses GPT-4o for essay analysis and XGBoost for classification.

**Contact:** Juan C. Rojas, MD, MS (juan_rojas@rush.edu)

## Quick Start for 2025 Admissions

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4o essay analysis)
- Excel file with applicant data ("1. Applicants.xlsx" format)

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd rmc_admissions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Production Pipeline

#### Option 1: Streamlit Web Interface (Recommended)
```bash
cd github_repo/web_app
streamlit run app.py
```
Then:
1. Upload your "1. Applicants.xlsx" file
2. System automatically processes essays with GPT-4o
3. Download results Excel file with predictions

#### Option 2: Command Line Script
```bash
cd scripts
python process_2025_applications.py --input "/path/to/1. Applicants.xlsx" --output "results_2025.xlsx"
```

## File Structure
```
rmc_admissions/
├── github_repo/
│   ├── web_app/           # Streamlit application
│   │   ├── app.py         # Main web interface
│   │   └── processor.py   # Application processing logic
│   ├── src/
│   │   ├── features/      # Feature engineering
│   │   │   ├── feature_engineer.py
│   │   │   └── essay_analyzer.py
│   │   └── models/        # Trained models
│   │       ├── cascade_classifier.py
│   │       └── cascade_model_2024.pkl
│   └── scripts/           # Production scripts
│       └── process_2025_applications.py
├── data/
│   └── processed/         # Where processed data is saved
├── models/
│   └── cascade_model_2024.pkl  # Trained model file
└── output/
    └── RMC_AI_Admissions_TRIPOD_Report_Final.html  # Documentation
```

## Data Requirements

### Input File Format: "1. Applicants.xlsx"
Required columns:
- `AMCAS ID` or `Amcas_ID`
- `Gender`
- `Date of Birth` or `Age`
- `State`
- `Citizenship`
- `First_Generation_Ind`
- `Disadvantanged_Ind`
- `Exp_Hour_Research`
- `Exp_Hour_Volunteer_Med`
- `Exp_Hour_Volunteer_Non_Med`
- `Comm_Service_Total_Hours`
- `HealthCare_Total_Hours`
- `military_service`

### Essay Requirements
- Essays should be in PDF format
- PDFs should be named with AMCAS ID (e.g., "15234567.pdf")
- Place PDFs in a folder accessible to the system

## Cost Estimates
- GPT-4o essay analysis: $0.15-0.25 per applicant
- Processing time: 5-10 seconds per applicant
- Can process hundreds of applications in parallel

## Output Format
The system generates an Excel file with:
- `AMCAS_ID`: Applicant identifier
- `Predicted_Quartile`: Q1 (Top 25%), Q2 (26-50%), Q3 (51-75%), Q4 (Bottom 25%)
- `Confidence_Score`: Model confidence (0-100%)
- `Predicted_Score`: Estimated reviewer score (0-25)
- `Essay_Scores`: 11 dimensions scored 0-100
- `Red_Flags`: Count of concerning elements
- `Green_Flags`: Count of exceptional elements

## Model Details
- **Algorithm**: XGBoost with 3-stage cascade architecture
- **Features**: 32 engineered features including demographics, experiences, and essay scores
- **Training Data**: 838 applicants from 2022-2023
- **Test Performance**: 93.8% accuracy on 613 applicants from 2024

## Troubleshooting

### Common Issues
1. **OpenAI API errors**: Check your API key is set correctly
2. **Missing data**: Model handles missing values, but ensure key columns exist
3. **PDF extraction fails**: Ensure PDFs are text-based, not scanned images

### Environment Variables
Set these before running:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Annual Maintenance
1. Retrain model with new admission cycle data
2. Update essay prompts if application requirements change
3. Validate predictions against human reviewer decisions
4. Check for demographic bias annually

## Support
For technical issues or questions:
- Contact: juan_rojas@rush.edu
- Documentation: See `output/RMC_AI_Admissions_TRIPOD_Report_Final.html`

---
Last Updated: December 2024
Version: 1.0 Production Release