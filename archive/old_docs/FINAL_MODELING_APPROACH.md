# Final Modeling Approach: Predicting Application Review Scores

## Key Findings

### 1. **Interview Threshold: Score ≥ 19** (NOT 18)
Based on actual data analysis:
- 2022-2023: Median score = 17
- 2024: Median score = 19
- **Recommendation**: Use score ≥ 19 as interview threshold (~50% selection rate)

### 2. **No Explicit Interview Column**
There is NO "Interview_Decision" column in the data. We must infer from Application Review Score.

## Comprehensive Feature Set

### A. Structured Features from Excel Files

#### 1. **Academic Features** (Continuous)
```python
# From Academic Records aggregation
- calculated_total_gpa         # Computed from transcript
- calculated_bcpm_gpa         # Biology/Chemistry/Physics/Math
- total_credit_hours
- course_diversity            # Number of unique subjects
- bcpm_percentage            # % of STEM courses

# From GPA Trends
- Total_GPA_Trend            # 0=declining, 1=improving
- BCPM_GPA_Trend            # 0=declining, 1=improving

# MCAT (needs to be added)
- MCAT_Total_Score
- MCAT_CPBS, MCAT_CARS, MCAT_BBLS, MCAT_PSBB
```

#### 2. **Experience Features** (Continuous)
```python
- Exp_Hour_Total              # Average ~9,200 hours
- Exp_Hour_Research           # Average ~1,100 hours
- Exp_Hour_Volunteer_Med      # Average ~600 hours
- Exp_Hour_Volunteer_Non_Med  # Average ~940 hours
- Exp_Hour_Employ_Med         # Average ~2,060 hours
- Exp_Hour_Shadowing          # Average ~150 hours
- Comm_Service_Total_Hours    # Average ~1,375 hours
- HealthCare_Total_Hours      # Average ~2,390 hours
```

#### 3. **Diversity & Holistic Features**
```python
# Binary indicators (convert Yes/No to 1/0)
- First_Generation_Ind        # ~16% are first-gen
- Disadvantanged_Ind         # Note: 2024 all "No"
- RU_Ind                     # Rural/Urban background
- Pell_Grant                 # Financial need indicator
- Fee_Assistance_Program
- Childhood_Med_Underserved_Self_Reported
- Family_Assistance_Program   # ~35% receive
- Paid_Employment_BF_18      # ~63% worked before 18

# From Language file aggregation
- num_languages              # Language diversity
- has_native_non_english
- language_diversity_score

# From Parent file aggregation  
- max_parent_education       # First-gen indicator
- parent_in_healthcare       # Medical family background
- single_parent_household

# From Sibling file
- num_siblings
- family_size
```

#### 4. **Categorical Features** (Need One-Hot Encoding)
```python
# Demographics
- Gender                     # M/F/Other/Unknown
- Citizenship               # 16 categories
- SES_Value                 # EO1/EO2

# Academic
- Major_Long_Desc           # Many categories
- Under_School              # Undergraduate institution

# Service
- Service_Rating_Categorical # 4 levels
- Service_Rating_Numerical   # 1-4 ordinal

# Income (19 categories - convert to ordinal)
- Family_Income_Level       # <$25K to $400K+
```

### B. Engineered Features

```python
# Experience ratios
- research_intensity = Research_Hours / Total_Hours
- clinical_intensity = (Clinical + Shadowing) / Total_Hours
- volunteer_intensity = Non_Med_Volunteer / Total_Hours

# Composite scores
- diversity_score = sum(all diversity indicators)
- service_commitment = Service_Rating × log(Service_Hours)
- age_adjusted_experience = Total_Hours / (Age - 18)

# Academic trajectory
- gpa_trend_score = Total_Trend + BCPM_Trend
- academic_trajectory = GPA × GPA_Trend
```

### C. LLM Features from Unstructured Text

```python
# Core evaluation dimensions (0-10 scale)
- llm_narrative_coherence      # Story quality
- llm_motivation_authenticity  # Genuine vs generic
- llm_reflection_depth        # Insight quality
- llm_growth_demonstrated     # Personal development
- llm_unique_perspective      # Distinctiveness
- llm_clinical_insight        # Healthcare understanding
- llm_service_genuineness     # Authentic commitment
- llm_leadership_impact       # Real impact shown
- llm_communication_quality   # Writing effectiveness
- llm_maturity_score         # Professional readiness

# Flags
- llm_red_flag_severity      # 0-10: Concerns
- llm_green_flag_strength    # 0-10: Exceptional qualities

# Overall
- llm_essay_overall_score    # 0-100: Essay quality
```

### D. Interaction Features

```python
# Academic × Essay
- academic_package = GPA × llm_academic_readiness

# Experience × Essay  
- clinical_readiness = Clinical_Hours × llm_clinical_insight
- research_quality = Research_Hours × llm_reflection_depth
- service_authenticity = Service_Rating × llm_service_genuineness

# Holistic score
- holistic_score = weighted_sum(academic, clinical, service, diversity, essay)
```

## Modeling Approaches

### 1. **Primary Model: XGBoost Regressor**
```python
# Predict continuous Application Review Score (0-25)
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    objective='reg:squarederror'
)
```

**Why XGBoost?**
- Handles mixed data types well
- Captures non-linear interactions
- Built-in feature importance
- Robust to outliers
- Can handle missing values

### 2. **Alternative Models to Test**

```python
# Random Forest (baseline)
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10
)

# Neural Network (for complex interactions)
nn_model = MLPRegressor(
    hidden_layers=(200, 100, 50),
    activation='relu',
    early_stopping=True,
    validation_fraction=0.15
)

# Ensemble (combine predictions)
ensemble = VotingRegressor([
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('nn', nn_model)
])
```

### 3. **Two-Stage Approach** (For Tier 2/3 distinction)

```python
# Stage 1: Predict score
predicted_score = model.predict(features)

# Stage 2: Refine around threshold
if 17 <= predicted_score <= 20:  # Near boundary
    # Use specialized model trained on boundary cases
    refined_score = boundary_model.predict(features)
```

## Training Strategy

### 1. **Data Split**
- **Train**: 2022 + 2023 combined (~900 samples)
- **Validation**: 20% of training data (stratified)
- **Test**: 2024 data (~400 samples)

### 2. **Cross-Validation**
```python
# Stratified K-fold to maintain score distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom splitter for time-based validation
time_split = TimeSeriesSplit(n_splits=3)
```

### 3. **Handling Class Imbalance**
For tier prediction, use:
- Sample weights based on tier frequency
- SMOTE for minority class oversampling
- Custom loss function penalizing Tier 2/3 errors

### 4. **Feature Selection**
```python
# Use SHAP values to identify top features
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Select top 50 most important features
feature_importance = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(feature_importance)[-50:]
```

## Evaluation Metrics

### 1. **For Continuous Score Prediction**
- Mean Absolute Error (MAE): Target < 2.0
- Root Mean Squared Error (RMSE): Target < 2.5
- R²: Target > 0.80
- Spearman correlation: Target > 0.85

### 2. **For Interview Decision (≥19)**
- Precision: % of predicted interviews that are correct
- Recall: % of actual interviews identified
- F1 Score: Balance of precision and recall
- AUC-ROC: Overall discrimination ability

### 3. **For Tier 2/3 Boundary**
- Confusion matrix around scores 17-20
- Precision/Recall specifically for boundary cases
- Cost-sensitive accuracy (penalize 2→3 errors more)

## Implementation Pipeline

```python
def train_reviewer_score_model():
    # 1. Load and merge all data sources
    engineer = ComprehensiveFeatureEngineer()
    df_2022 = engineer.load_and_merge_all_data("data", 2022)
    df_2023 = engineer.load_and_merge_all_data("data", 2023)
    df_train = pd.concat([df_2022, df_2023])
    
    # 2. Load LLM scores
    llm_scores = load_llm_evaluations()  # From Azure OpenAI
    
    # 3. Prepare features
    X_train, feature_names = engineer.prepare_final_features(
        df_train, llm_scores, fit_scaler=True
    )
    
    # 4. Create target
    y_train = df_train['Application Review Score']
    
    # 5. Train model with cross-validation
    model = XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_mean_absolute_error')
    
    # 6. Final training
    model.fit(X_train, y_train)
    
    # 7. Evaluate on 2024 test set
    df_2024 = engineer.load_and_merge_all_data("data", 2024)
    X_test, _ = engineer.prepare_final_features(df_2024, fit_scaler=False)
    y_test = df_2024['Application Review Score']
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Test MAE: {mae:.2f} points")
    print(f"Interview accuracy: {interview_accuracy:.1%}")
    
    return model
```

## Expected Performance

Based on feature richness and similar medical admissions studies:

1. **Score Prediction**: MAE of 1.8-2.2 points (7-9% of scale)
2. **Interview Decision**: 85-90% accuracy
3. **Tier Classification**: 75-80% exact, 92-95% adjacent
4. **Tier 2/3 Boundary**: 70-75% precision

The combination of comprehensive structured features and LLM-evaluated essay quality should provide strong predictive power for reviewer scores.