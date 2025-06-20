# Model Files

This directory contains the trained AI models for the admissions system.

## Model Files

The trained models are stored as `.pkl` files and are not included in the repository due to their size.

### Downloading Models

To download the pre-trained models:

```bash
python scripts/download_model.py
```

### Model Naming Convention

Models are named with the following pattern:
```
{model_type}_{timestamp}.pkl
```

Example: `comprehensive_cascade_20250619_193352.pkl`

### Model Contents

Each `.pkl` file contains:
- `optimizer`: The trained cascade classifier
- `feature_cols`: List of features used
- `imputer`: Sklearn imputer for missing values
- `scaler`: Sklearn scaler for normalization
- `performance`: Model performance metrics
- `hyperparameters`: Optimized parameters

### Model Versions

| Version | Date | Accuracy | Notes |
|---------|------|----------|-------|
| v1.0 | 2025-06-19 | 80.8% | Initial cascade model |

### Using Custom Models

To use a specific model file:

```python
from src.processors import ApplicationProcessor

processor = ApplicationProcessor(model_path='models/your_model.pkl')
```

### Model Size

Typical model files are 50-200 MB depending on the ensemble size.

## Security Note

Do not commit model files to the repository. They should be stored securely and distributed separately.