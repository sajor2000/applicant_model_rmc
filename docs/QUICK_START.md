# Quick Start Guide

This guide will help you get the Rush AI Admissions System up and running quickly.

## Prerequisites

- Python 3.9 or higher
- Azure OpenAI API access
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RushUniversity/ai-admissions.git
cd ai-admissions
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your Azure OpenAI credentials
# Required values:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
```

### 5. Download Model

```bash
# Download the pre-trained model
python scripts/download_model.py

# Or place your model file in models/
```

## Basic Usage

### Command Line Interface

Process a single application:
```bash
python -m rush_admissions process --file sample_application.csv
```

Batch processing:
```bash
python -m rush_admissions batch --input applications/ --output results.xlsx
```

### Web Application

Launch the web interface:
```bash
streamlit run web_app/app.py
```

Then open http://localhost:8501 in your browser.

Demo credentials:
- Username: `rush_admin`
- Password: `demo2025`

## Sample Data Format

Your CSV/Excel file should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| amcas_id | Unique identifier | Yes |
| service_rating_numerical | Service rating (1-4) | Yes |
| healthcare_total_hours | Clinical experience hours | Yes |
| essay_text | Combined essay text | Yes |
| age | Applicant age | Yes |
| gender | Male/Female/Other | No |
| first_generation_ind | 0 or 1 | No |

See `data/sample/sample_applications.csv` for a complete example.

## Processing Flow

1. **Upload** your data file
2. **Process** through the AI model
3. **Review** results with confidence scores
4. **Export** to Excel/CSV

## Results Interpretation

- **Q1**: Top 25% - Strong candidates for interview
- **Q2**: 50-75% - Above average candidates
- **Q3**: 25-50% - Below average candidates  
- **Q4**: Bottom 25% - Weakest applications

Confidence scores:
- 🟢 80%+: High confidence (auto-process)
- 🟡 60-80%: Medium confidence (quick review)
- 🔴 <60%: Low confidence (detailed review)

## Troubleshooting

### "Model not found" error
- Run `python scripts/download_model.py`
- Check `models/` directory exists

### "API key not found" error
- Verify `.env` file exists
- Check Azure credentials are correct

### Processing takes too long
- Essay analysis via GPT-4o takes ~2 seconds per application
- Batch processing is more efficient

## Next Steps

- Read the [Full Documentation](docs/README.md)
- Review [Feature Dictionary](docs/FEATURE_DICTIONARY.md)
- See [API Reference](docs/API_REFERENCE.md) for programmatic use

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/RushUniversity/ai-admissions/issues)
- Email: admissions-ai@rush.edu