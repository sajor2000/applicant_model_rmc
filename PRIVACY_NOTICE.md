# Privacy Notice

This repository has been audited for privacy and security. The following measures are in place:

## ✅ Privacy Protected

1. **No Real Applicant Data**: All CSV/Excel files with real applicant information are excluded via .gitignore
2. **Synthetic Sample Data Only**: The `data/sample/` directory contains only fake/synthetic examples
3. **No Personal Emails**: All personal email addresses have been removed
4. **No API Keys**: All API keys and credentials are managed via environment variables
5. **Model Files**: Pickle files contain only trained model parameters, no training data

## 🛡️ Security Measures

### Environment Variables
- All sensitive configuration stored in `.env` files
- Comprehensive `.gitignore` prevents accidental commits
- Pattern matching for all env file variants

### Data Protection
- Real data directories (`data/raw/*`, `data/processed/*`) are git-ignored
- Only anonymized sample data in repository
- No AMCAS IDs or personal identifiers

## 📋 For Contributors

Before committing:
1. Never commit real applicant data
2. Use environment variables for all credentials
3. Anonymize any examples or test data
4. Remove personal contact information

## 🔒 Model Security

The model pickle files contain:
- Trained model parameters only
- Feature names and preprocessing pipelines
- No training data or personal information
- Safe for distribution

Last Privacy Audit: June 20, 2025