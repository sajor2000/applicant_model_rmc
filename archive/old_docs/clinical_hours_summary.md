# Clinical Experience Hours in Medical Admissions Data

## Summary
Clinical experience hours are already available in the standardized applicants data. They are NOT missing - they are stored in multiple columns in the `1. Applicants.xlsx` file.

## Available Clinical Hours Columns in Applicants File

### Direct Clinical Experience Columns:
1. **`exp_hour_volunteer_med`** (Medical/Clinical Volunteering)
   - Non-null: 396/437 applicants (90.6%)
   - Mean: 643.58 hours
   - Range: 16 - 9,650 hours

2. **`exp_hour_shadowing`** (Physician Shadowing)
   - Non-null: 379/437 applicants (86.7%)
   - Mean: 184.44 hours
   - Range: 4 - 6,760 hours

3. **`exp_hour_employ_med`** (Medical/Clinical Employment)
   - Non-null: 318/437 applicants (72.8%)
   - Mean: 2,637.29 hours
   - Range: 30 - 16,000 hours

4. **`healthcare_total_hours`** (Total Healthcare Experience)
   - Non-null: 437/437 applicants (100%)
   - Mean: 2,662.28 hours
   - Range: 52 - 17,860 hours
   - **This appears to be the aggregated total of all clinical experiences**

### Related Experience Columns:
- `exp_hour_total` - Total of all experience hours
- `exp_hour_research` - Research hours
- `exp_hour_volunteer_non_med` - Non-medical volunteering
- `comm_service_total_hours` - Community service hours

## Alternative: Calculate from Experiences File

The `6. Experiences.xlsx` file contains detailed experience entries that can be aggregated:

### Clinical Experience Types in Experiences File:
- Community Service/Volunteer - Medical/Clinical (799 entries)
- Physician Shadowing/Clinical Observation (497 entries)
- Paid Employment - Medical/Clinical (479 entries)

## Current Usage in Models

The feature engineering pipelines are already using these columns:

1. **comprehensive_feature_engineering.py** includes:
   - `Exp_Hour_Volunteer_Med`
   - `Exp_Hour_Shadowing`
   - `Exp_Hour_Employ_Med`
   - `HealthCare_Total_Hours`

2. **data_transformation_pipeline.py** includes:
   - Creates `clinical_proportion` from volunteer_med + shadowing hours
   - Handles missing values by filling with 0

## Recommendations

1. **Clinical hours are NOT missing** - they are available in multiple columns
2. **Use `healthcare_total_hours`** as the primary clinical experience metric (100% complete)
3. **For detailed analysis**, can use the individual components:
   - Medical volunteering hours
   - Shadowing hours  
   - Medical employment hours
4. **For modeling**, the current pipelines are already incorporating these features correctly

## Column Name Standardization

Note: The `fix_column_names.py` script standardizes these to lowercase with underscores:
- `Exp_Hour_Volunteer_Med` → `exp_hour_volunteer_med`
- `HealthCare_Total_Hours` → `healthcare_total_hours`
- etc.

This ensures consistency across different years of data.