# Rush Medical College AI Admissions System
# 2025 Deployment Checklist

## Pre-Deployment Setup

### 1. Environment Setup
- [ ] Install Python 3.8 or higher
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`

### 2. API Configuration
- [ ] Obtain OpenAI API key for GPT-4o
- [ ] Create `.env` file from `.env.example`
- [ ] Add API key to `.env` file
- [ ] Test API connection

### 3. Model Verification
- [ ] Verify `models/cascade_model_2024.pkl` exists
- [ ] Run test prediction on sample data
- [ ] Confirm 93.8% accuracy on test set

## Production Deployment

### Option A: Streamlit Web Interface (Recommended)
1. Navigate to web app directory:
   ```bash
   cd github_repo/web_app
   ```

2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Access at: http://localhost:8501

### Option B: Command Line Processing
1. Navigate to scripts directory:
   ```bash
   cd scripts
   ```

2. Run batch processing:
   ```bash
   python process_2025_applications.py \
     --input "/path/to/1. Applicants.xlsx" \
     --output "results_2025.xlsx" \
     --essays "/path/to/essay/pdfs"
   ```

## Data Preparation

### Required Excel Columns
- [ ] AMCAS ID or Amcas_ID
- [ ] Gender
- [ ] Date of Birth or Age
- [ ] State
- [ ] Citizenship
- [ ] First_Generation_Ind
- [ ] Disadvantanged_Ind
- [ ] Experience hours (5 types)
- [ ] military_service

### Essay PDFs
- [ ] PDFs named by AMCAS ID (e.g., "15234567.pdf")
- [ ] All PDFs in single directory
- [ ] Text-based PDFs (not scanned images)

## Quality Assurance

### Pre-Processing Checks
- [ ] Verify all required columns present
- [ ] Check for data type consistency
- [ ] Validate AMCAS ID format
- [ ] Ensure essay PDFs are readable

### Post-Processing Validation
- [ ] Review confidence scores distribution
- [ ] Check quartile distribution matches expectations
- [ ] Verify no processing errors in logs
- [ ] Sample check 10 random predictions

## Production Monitoring

### Performance Metrics
- [ ] Processing time per applicant < 10 seconds
- [ ] GPT-4o cost per applicant < $0.25
- [ ] Model confidence > 90% for most predictions
- [ ] Error rate < 1%

### Weekly Reviews
- [ ] Compare AI predictions to human decisions
- [ ] Monitor for demographic bias
- [ ] Track processing costs
- [ ] Update documentation as needed

## Support Resources

- **Technical Contact**: [Contact Email Removed]
- **Documentation**: output/RMC_AI_Admissions_TRIPOD_Report_Final.html
- **GitHub Repository**: [Add your repository URL]
- **Emergency Support**: [Add contact info]

## Annual Maintenance (Post-2025 Cycle)

- [ ] Collect 2025 human reviewer decisions
- [ ] Retrain model with 2025 data
- [ ] Validate improved accuracy
- [ ] Update documentation
- [ ] Archive 2024 model
- [ ] Deploy 2025 model for 2026 cycle

---
Last Updated: December 2024
Version: 1.0