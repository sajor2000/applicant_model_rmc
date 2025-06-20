# Rush Medical College AI Admissions System

<p align="center">
  <img src="docs/images/rush_logo.png" alt="Rush Medical College" width="300"/>
</p>

<p align="center">
  <strong>An AI-powered system to help identify the most promising future physicians from thousands of applications</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#performance">Performance</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#web-application">Web Application</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

The Rush Medical College AI Admissions System combines traditional application metrics with advanced essay analysis using GPT-4o to provide fair, accurate, and efficient applicant ranking. The system helps admissions committees focus their expertise on borderline cases while maintaining high accuracy in identifying top candidates.

### Key Features

- **80.8% Accuracy** in quartile placement
- **99% Near-Perfect Accuracy** (within one quartile)
- **Zero Bias** across gender, socioeconomic status, and age
- **73 Features Analyzed** including essays via GPT-4o
- **Confidence Scoring** to identify cases needing human review

## Performance

| Metric | Performance | Details |
|--------|-------------|---------|
| Exact Match | 80.8% | 4 out of 5 applicants correctly ranked |
| Adjacent Accuracy | 99.0% | Only 1% misplaced by more than one quartile |
| Top Talent Detection | 91.7% | Correctly identifies strongest candidates |
| Model AUC | 0.945 | Excellent discrimination between strong/weak |

## Installation

### Prerequisites

- Python 3.9 or higher
- Azure OpenAI API access
- 8GB RAM minimum

### ⚠️ Privacy Notice

This repository contains **ONLY synthetic test data**. Never commit real applicant information.

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rush-ai-admissions.git
cd rush-ai-admissions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

4. Download the pre-trained model:
```bash
python scripts/download_model.py
```

## Usage

### Command Line Interface

Process a single application:
```bash
python src/process_application.py --file application.csv
```

Batch processing:
```bash
python src/batch_process.py --input-dir applications/ --output results.xlsx
```

### Web Application

Launch the interactive web interface:
```bash
streamlit run web_app/app.py
```

Access at `http://localhost:8501`

<p align="center">
  <img src="docs/images/webapp_screenshot.png" alt="Web Application Screenshot" width="600"/>
</p>

## Project Structure

```
rush-ai-admissions/
├── src/                    # Core processing code
│   ├── models/            # Model training and inference
│   ├── features/          # Feature engineering
│   └── processors/        # Application processing
├── web_app/               # Streamlit web application
├── data/                  # Sample data (anonymized)
├── models/                # Trained model files
├── notebooks/             # Jupyter notebooks for analysis
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── scripts/               # Utility scripts
```

## How It Works

1. **Data Collection**: Structured application data (GPA, hours, demographics)
2. **Essay Analysis**: GPT-4o evaluates essays for authenticity, maturity, and insight
3. **Feature Engineering**: Creates interaction features and coherence scores
4. **Ranking**: XGBoost cascade classifier assigns quartiles with confidence
5. **Review Routing**: Low-confidence cases flagged for human review

## Documentation

- [Full Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Feature Dictionary](docs/FEATURE_DICTIONARY.md)
- [Model Architecture](docs/MODEL_ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

## Fairness & Ethics

The system has been rigorously tested for bias:
- **Gender**: No significant differences (p = 0.976)
- **Socioeconomic**: Equal representation across quartiles
- **Age**: Minimal correlation with outcomes
- **International**: Fair evaluation regardless of background

See our [Fairness Report](docs/FAIRNESS_REPORT.md) for details.

## Results Visualization

View our [Interactive Results Dashboard](https://yourusername.github.io/rush-ai-admissions/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:
```bibtex
@software{rush_ai_admissions_2025,
  title = {Rush Medical College AI Admissions System},
  author = {Rush University Medical Center},
  year = {2025},
  url = {https://github.com/yourusername/rush-ai-admissions}
}
```

## Contact

For questions or support:
- Technical Issues: [Open an issue](https://github.com/yourusername/rush-ai-admissions/issues)
- General Inquiries: [Contact Information Removed]

---

<p align="center">
  Developed with integrity and tested for fairness at Rush University Medical Center
</p>