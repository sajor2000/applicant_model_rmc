# Rush Medical College AI Admissions Assistant - Web Application

## Overview
This Streamlit web application provides a user-friendly interface for the Rush Medical College admissions team to process applications using the trained AI model.

## Features
- 🔐 Secure login system
- 📤 Batch application processing
- 📊 Real-time results visualization
- 📈 Analytics dashboard
- 📥 Export to Excel/CSV
- 🎯 Confidence-based prioritization

## Installation

### 1. Install Requirements
```bash
pip install -r requirements_app.txt
```

### 2. Configure Environment
Ensure your `.env` file contains:
```
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### 3. Ensure Model Files
The app expects trained model files in the `models/` directory. The latest cascade model will be loaded automatically.

## Running the Application

### Local Development
```bash
streamlit run admissions_app.py
```

The app will open at `http://localhost:8501`

### Demo Credentials
- Username: `rush_admin`
- Password: `demo2025`

## Usage Guide

### 1. Dashboard
- View model performance metrics
- See quick statistics
- Access all main features

### 2. Process Applications
1. Upload Excel/CSV file with application data
2. Required fields:
   - `amcas_id` - Unique identifier
   - `service_rating_numerical` - Service rating (1-4)
   - `healthcare_total_hours` - Clinical experience
   - `essay_text` - Combined essay text
   - Additional fields as per model requirements

3. Click "Process Applications"
4. View real-time progress
5. See summary statistics

### 3. View Results
- Filter by quartile, confidence, review status
- Search by ID
- View detailed applicant information
- Color-coded confidence levels:
  - 🟢 Green: High confidence (≥80%)
  - 🟡 Yellow: Medium confidence (60-80%)
  - 🔴 Red: Low confidence (<60%)

### 4. Analytics
- Gender distribution analysis
- Confidence breakdowns
- Review needs by quartile
- Feature importance insights

### 5. Export
- Excel format with multiple sheets
- CSV for data analysis
- Filtered exports (Q1 only, need review, etc.)

## File Structure
```
admissions_app.py          # Main Streamlit application
app_processor.py          # Application processing logic
app_results.py           # Results display module
requirements_app.txt     # Python dependencies
```

## Data Format

### Input File Structure
Your Excel/CSV should contain:
```
amcas_id | service_rating_numerical | healthcare_total_hours | essay_text | ...
12345    | 4                       | 500                    | "Essay..." | ...
```

### Output Format
The system provides:
- `predicted_quartile`: Q1 (top) to Q4 (bottom)
- `confidence`: 0-100% confidence score
- `needs_review`: Boolean flag for human review
- Probability breakdowns for each outcome

## Deployment Options

### Option 1: Local Server
Run on a dedicated machine at Rush Medical College with access restricted to admissions network.

### Option 2: Azure App Service
Deploy to Azure for cloud-based access:

1. Create Azure App Service
2. Configure with Python 3.9+
3. Deploy code via Git
4. Set environment variables
5. Configure authentication

### Option 3: Streamlit Cloud
For quick deployment:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with environment variables

## Security Considerations

1. **Authentication**: Implement proper SSO/LDAP integration
2. **Data Protection**: Ensure HIPAA compliance
3. **Access Logs**: Track all user actions
4. **Encryption**: Use HTTPS for all connections
5. **Data Retention**: Follow institutional policies

## Troubleshooting

### Model Not Found
- Ensure model files are in `models/` directory
- Check file permissions

### API Connection Failed
- Verify Azure OpenAI credentials in `.env`
- Check network connectivity

### Processing Errors
- Validate input data format
- Check for missing required fields
- Review error logs

## Support
For technical support, contact the AI development team.

## Version
Current Version: 1.0
Last Updated: June 2025