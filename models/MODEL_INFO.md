# Model Information

## Available Models

### production_model_93_8_accuracy.pkl (MAIN PRODUCTION MODEL)
- **Training Data**: Combined 2022 and 2023 admission cycles
- **Model Type**: Refined Cascade Classifier with GPT-4o integration
- **Accuracy**: 93.8% on 2024 holdout test set (38 errors out of 613)
- **Adjacent Accuracy**: 100% (no predictions off by more than 1 quartile)
- **Purpose**: Production model for 2025+ admissions screening
- **Features**: 
  - Demographics (age, gender, citizenship, first_generation)
  - Experience hours (clinical, research, volunteer)
  - Service rating
  - GPT-4o essay analysis scores (11 dimensions)
- **File Size**: ~1.7 MB
- **Note**: This is the same as refined_gpt4o_latest.pkl

### refined_gpt4o_latest.pkl
- **Training Data**: 2022-2023 data with GPT-4o features
- **Model Type**: Refined model with GPT-4o integration
- **Purpose**: Enhanced predictions with essay analysis
- **File Size**: ~1.7 MB

## Usage

```python
import pickle

# Load the cascade model
with open('models/cascade_model_2022_2023.pkl', 'rb') as f:
    model_data = pickle.load(f)

# The pickle file contains:
# - 'model': The trained XGBoost classifier
# - 'feature_columns': List of required features
# - 'scaler': StandardScaler for preprocessing
# - 'imputer': SimpleImputer for missing values
```

## Privacy Note
These model files contain only:
- Trained model parameters
- Feature names and preprocessing pipelines
- No training data or personal information
- Safe for distribution and deployment