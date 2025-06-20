# Repository Structure

This document explains the organization of the Rush AI Admissions System repository.

## Directory Layout

```
rmc_admissions/
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model classes
│   │   ├── __init__.py
│   │   ├── cascade_classifier.py
│   │   └── model_loader.py
│   ├── processors/             # Application processing
│   │   ├── __init__.py
│   │   └── application_processor.py
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_engineer.py
│   │   └── essay_analyzer.py
│   └── utils/                  # Utility functions
│       └── __init__.py
│
├── web_app/                    # Streamlit web application
│   ├── app.py                  # Main application
│   ├── processor.py            # Processing logic
│   ├── results.py              # Results display
│   └── README.md
│
├── data/                       # Data directory
│   ├── 2022 Applicants Reviewed by Trusted Reviewers/
│   ├── 2023 Applicants Reviewed by Trusted Reviewers/
│   ├── 2024 Applicants Reviewed by Trusted Reviewers/
│   ├── processed/              # Processed data (empty)
│   ├── raw/                    # Raw data (empty)
│   ├── sample/                 # Sample data files
│   │   ├── sample_applications.csv
│   │   └── README.md
│   └── README.md
│
├── data_filtered/              # Filtered applicant data
│   ├── 2022_filtered_applicants.xlsx
│   ├── 2023_filtered_applicants.xlsx
│   └── 2024_filtered_applicants.xlsx
│
├── data_standardized/          # Standardized column names
│   ├── 2022_standardized/
│   ├── 2023_standardized/
│   └── 2024_standardized/
│
├── models/                     # Trained models
│   ├── production_model_93_8_accuracy.pkl  # Main production model
│   ├── refined_gpt4o_20250620_063117.pkl  # Latest refined model
│   ├── checkpoints/            # Model checkpoints (empty)
│   ├── trained/                # Trained models (empty)
│   ├── MODEL_INFO.md          # Model information
│   └── README.md              # Model download instructions
│
├── docs/                       # Documentation
│   ├── index.html             # Interactive dashboard
│   ├── confusion_matrix_2024.png
│   ├── performance_summary_2024.png
│   ├── images/                # Documentation images
│   ├── reports/               # Generated reports
│   ├── QUICK_START.md         # Getting started guide
│   └── README.md
│
├── output/                     # Model outputs and reports
│   ├── evaluation_2024_20250620_063608/
│   │   ├── confusion_matrix_2024.png
│   │   ├── performance_summary_2024.png
│   │   └── evaluation_report_2024.txt
│   ├── RMC_AI_Admissions_TRIPOD_Report.html
│   └── RMC_AI_Admissions_TRIPOD_Report_Final.html
│
├── archive/                    # Archived files
│   ├── old_docs/              # Previous documentation versions
│   ├── old_models/            # Previous model versions
│   ├── old_outputs/           # Previous outputs and results
│   └── old_scripts/           # Previous script versions
│
├── scripts/                    # Utility scripts
│   ├── analyze_data_quality.py
│   ├── analyze_transformations.py
│   ├── cleanup_repository.py
│   ├── download_model.py
│   ├── evaluate_2024_holdout.py
│   ├── find_academic_metrics.py
│   ├── generate_gpt4o_scores.py
│   ├── generate_synthetic_data.py
│   ├── optimize_gpt4o_complete.py
│   ├── optimize_gpt4o_final.py
│   ├── process_2025_applications.py
│   ├── process_applications.py
│   └── simulate_gpt4o_scores.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory/           # Exploratory analysis
│   └── training/              # Model training notebooks
│
├── tests/                      # Test suite
│   ├── test_processor.py      # Processor tests
│   ├── test_basic.py          # Basic functionality tests
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
│
├── logs/                       # Log files
│   └── gpt4o_optimization_*.log
│
├── deployment/                 # Deployment resources
│   ├── deploy_instructions.md
│   ├── deploy_to_github.sh
│   ├── quick_deploy_guide.md
│   └── rmc_ai_admissions_report/
│
├── github_repo/               # GitHub repository template
│   └── [Repository structure for GitHub deployment]
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── LICENSE                    # MIT License
├── README.md                  # Main documentation
├── README_PRODUCTION.md       # Production deployment guide
├── CONTRIBUTING.md            # Contribution guidelines
├── PRIVACY_NOTICE.md          # Privacy information
├── PRIVACY_AUDIT_REPORT.md    # Privacy audit results
├── DEPLOYMENT_CHECKLIST_2025.md  # Deployment checklist
├── GITHUB_SETUP.md            # GitHub setup instructions
├── CLEANUP_SUMMARY.md         # Cleanup documentation
├── REPOSITORY_STRUCTURE.md    # This file
└── rmc_admissions.code-workspace  # VS Code workspace
```

## Key Components

### 1. Source Code (`src/`)
Core Python modules organized by functionality:
- **models/**: Machine learning model implementations
- **processors/**: Application processing logic
- **features/**: Feature engineering pipeline
- **utils/**: Helper functions and utilities

### 2. Web Application (`web_app/`)
Streamlit-based user interface for non-technical users:
- Interactive dashboard
- File upload and processing
- Results visualization
- Export functionality

### 3. Documentation (`docs/`)
Comprehensive documentation:
- Interactive HTML dashboard
- Performance visualizations
- Quick start guide
- Technical documentation

### 4. Data Management
- **data/**: Original data files by year
- **data_filtered/**: Filtered applicant data
- **data_standardized/**: Data with standardized column names
- **data/sample/**: Sample data for testing

### 5. Models (`models/`)
Trained model storage:
- Production model with 93.8% accuracy
- Model versioning information
- Download instructions for large models

### 6. Output and Reports (`output/`)
- Model evaluation results
- TRIPOD compliance reports
- Performance metrics and visualizations

### 7. Archive (`archive/`)
Previous versions organized by type:
- Old documentation
- Previous model versions
- Historical outputs
- Legacy scripts

## Development Workflow

1. **Setup**: Clone repo, create virtual environment, install dependencies
2. **Development**: Make changes in feature branches
3. **Testing**: Run test suite before commits
4. **Documentation**: Update docs with changes
5. **Review**: Create pull request for review
6. **Deploy**: Merge to main triggers deployment

## Configuration Files

- **requirements.txt**: Production dependencies
- **setup.py**: Package configuration for pip install
- **.gitignore**: Excludes sensitive/large files
- **rmc_admissions.code-workspace**: VS Code workspace settings

## Security Considerations

- No real applicant data in repository
- API keys in environment variables only
- Model files distributed separately
- Regular privacy audits conducted

## Maintenance Notes

- Archive old files to `archive/` directory
- Keep `output/` directory clean - only latest results
- Remove Python cache files regularly
- Update documentation after major changes

## Getting Help

- See [QUICK_START.md](docs/QUICK_START.md) for setup
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development
- Review [README_PRODUCTION.md](README_PRODUCTION.md) for deployment
- Open issues for bugs/features