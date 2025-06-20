# Privacy Audit Report - RMC Admissions Repository
Date: 2025-06-20

## Summary of Findings

### 1. Personal Information Found

#### Email Addresses
- **[Contact Email Removed]** appears in multiple files:
  - GITHUB_SETUP.md
  - DEPLOYMENT_CHECKLIST_2025.md
  - README_PRODUCTION.md
  - scripts/process_2025_applications.py
  
**Recommendation**: Replace with generic email like "admin@institution.edu" or "contact@example.com"

### 2. Sensitive Data in CSV Files

#### AMCAS IDs
- Real AMCAS IDs (8-digit numbers) found in:
  - data/2022 Applicants Reviewed by Trusted Reviewers/llm_scores_2022.csv
  - data/2023 Applicants Reviewed by Trusted Reviewers/llm_scores_2023.csv
  - data/2024 Applicants Reviewed by Trusted Reviewers/llm_scores_2024.csv
  - Various output CSV files in archive/old_outputs/
  - output/candidate_rankings_*.csv files

**Recommendation**: These appear to be real applicant identifiers and should be anonymized or removed

### 3. API Keys and Credentials
- No hardcoded API keys found
- Proper use of environment variables (os.getenv) for API keys
- .env.example file contains placeholder values only

**Status**: GOOD - No security issues found

### 4. Medical/Patient Information
- No actual patient data or medical record numbers found
- HTML reports contain only generic references to "patient care" in educational context

**Status**: GOOD - No PHI found

### 5. Institutional-Specific Information

#### References to Rush Medical College
Found in multiple files:
- Python files reference "Rush Medical College" and "Rush University"
- Documentation refers to "Rush" specifically
- Archive contains files with "RUSH" in filenames

**Recommendation**: Consider generalizing to "Medical College" or "[Institution Name]"

### 6. Data Directories

#### Excel Files
The data directories contain extensive applicant data in Excel format:
- 1. Applicants.xlsx
- 2. Language.xlsx
- 3. Parents.xlsx
- 4. Siblings.xlsx
- 5. Academic Records.xlsx
- 6. Experiences.xlsx
- 8. Schools.xlsx
- 9. Personal Statement.xlsx
- 10. Secondary Application.xlsx
- 11. Military.xlsx
- 12. GPA Trend.xlsx

**Critical Issue**: These files likely contain real applicant data including:
- Personal statements
- Family information
- Academic records
- Secondary application essays

**Recommendation**: These files should NOT be in a public repository

### 7. Model Pickle Files
- Multiple .pkl files found in archive/old_models/ and models/
- Cannot inspect contents without loading, but these may contain:
  - Training data samples
  - Feature names that could reveal sensitive information
  
**Recommendation**: Verify these don't contain actual applicant data

## Critical Actions Required

1. **REMOVE all Excel files** from data directories - these contain real applicant information
2. **ANONYMIZE all AMCAS IDs** in CSV files or remove these files entirely
3. **REPLACE [Contact Email Removed]** with generic email throughout repository
4. **GENERALIZE Rush Medical College** references to make repository institution-agnostic
5. **VERIFY pickle files** don't contain training data samples with real information

## Files/Directories to Remove or Sanitize

### Must Remove:
- data/2022 Applicants Reviewed by Trusted Reviewers/*.xlsx
- data/2023 Applicants Reviewed by Trusted Reviewers/*.xlsx
- data/2024 Applicants Reviewed by Trusted Reviewers/*.xlsx
- data_filtered/*.xlsx
- data_standardized/*/*.xlsx

### Must Sanitize:
- All CSV files containing AMCAS IDs
- All files containing [Contact Email Removed]
- All files with Rush-specific references

## Verification Steps

After cleanup:
1. Re-run grep searches for email patterns
2. Verify no Excel files remain in data directories
3. Check all CSV files have anonymized IDs
4. Confirm no institution-specific information remains