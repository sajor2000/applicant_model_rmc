# Repository Structure

This document explains the organization of the Rush AI Admissions System repository.

## Directory Layout

```
rush-ai-admissions/
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model classes
│   │   ├── cascade_classifier.py
│   │   └── model_loader.py
│   ├── processors/             # Application processing
│   │   ├── application_processor.py
│   │   └── batch_processor.py
│   ├── features/               # Feature engineering
│   │   ├── feature_engineer.py
│   │   └── essay_analyzer.py
│   └── utils/                  # Utility functions
│       ├── data_validation.py
│       └── metrics.py
│
├── web_app/                    # Streamlit web application
│   ├── app.py                  # Main application
│   ├── processor.py            # Processing logic
│   ├── results.py              # Results display
│   └── README.md
│
├── data/                       # Data directory
│   ├── sample/                 # Sample data files
│   │   └── sample_applications.csv
│   └── README.md
│
├── models/                     # Model storage
│   └── README.md              # Model download instructions
│
├── docs/                       # Documentation
│   ├── index.html             # Interactive dashboard
│   ├── QUICK_START.md         # Getting started guide
│   ├── TECHNICAL_DOCS.md      # Technical documentation
│   ├── FEATURE_DICTIONARY.md   # Feature definitions
│   ├── MODEL_ARCHITECTURE.md   # Model details
│   ├── API_REFERENCE.md       # API documentation
│   └── DEPLOYMENT_GUIDE.md    # Deployment instructions
│
├── notebooks/                  # Jupyter notebooks
│   ├── model_training.ipynb   # Training process
│   ├── feature_analysis.ipynb # Feature importance
│   └── fairness_testing.ipynb # Bias analysis
│
├── tests/                      # Test suite
│   ├── test_processor.py      # Processor tests
│   ├── test_features.py       # Feature engineering tests
│   ├── test_models.py         # Model tests
│   └── conftest.py            # Test configuration
│
├── scripts/                    # Utility scripts
│   ├── download_model.py      # Model downloader
│   ├── validate_data.py       # Data validation
│   └── generate_report.py     # Report generation
│
├── .github/                    # GitHub configuration
│   └── workflows/             # CI/CD pipelines
│       ├── tests.yml          # Run tests
│       └── deploy-docs.yml    # Deploy documentation
│
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package setup
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # Contribution guidelines
└── REPOSITORY_STRUCTURE.md   # This file
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
- User guides
- Technical documentation
- API reference
- Deployment instructions

### 4. Tests (`tests/`)
Test coverage for all components:
- Unit tests
- Integration tests
- Mock data fixtures

### 5. Data (`data/`)
Sample data and data specifications:
- Example CSV files
- Data format documentation
- No real applicant data (privacy)

### 6. Models (`models/`)
Trained model storage:
- Model files not in repository (too large)
- Download instructions
- Model versioning information

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
- **.env.example**: Template for environment variables
- **.gitignore**: Excludes sensitive/large files

## CI/CD

GitHub Actions workflows:
- Automated testing on push/PR
- Documentation deployment to GitHub Pages
- Code quality checks (linting, type checking)

## Security Considerations

- No real applicant data in repository
- API keys in environment variables only
- Model files distributed separately
- Regular security audits

## Getting Help

- See [QUICK_START.md](docs/QUICK_START.md) for setup
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development
- Open issues for bugs/features
- Email: [Contact Information Removed]