
## Recommendations to Reduce Low Confidence Cases

### Immediate Improvements (1-2 days):

1. **Increase Model Complexity**
   - Current: 300 trees → Recommended: 500-700 trees
   - Current: Single model → Recommended: 5-model ensemble
   - Expected impact: 20-30% reduction in low confidence

2. **Probability Calibration**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   
   # Wrap each stage classifier
   calibrated_model = CalibratedClassifierCV(
       base_estimator=xgb_model,
       method='isotonic',  # or 'sigmoid'
       cv=3
   )
   ```
   - Expected impact: 15-20% reduction in borderline cases

3. **Optimize Confidence Thresholds**
   - Analyze human review outcomes
   - Set thresholds to minimize false low-confidence
   - Consider quartile-specific thresholds

### Advanced Improvements (1 week):

1. **Add Confidence-Specific Features**
   ```python
   # Profile coherence score
   coherence_features = {
       'essay_structured_alignment': correlation(essay_scores, structured_scores),
       'internal_consistency': std(feature_percentiles),
       'profile_completeness': 1 - (missing_features / total_features),
       'extreme_feature_count': sum(features > 95th_percentile)
   }
   ```

2. **Multi-Stage Confidence**
   - Stage 1 confidence: Clear reject vs borderline
   - Stage 2 confidence: Clear middle vs Q1/Q3 border
   - Stage 3 confidence: Clear accept vs Q2/Q3 border
   - Combine for overall confidence

3. **Active Learning Integration**
   - Track which low-confidence cases humans overturn
   - Retrain on these specific patterns
   - Build "ambiguity detector" model

### Expected Outcomes:

Current: 21% low confidence (129/613)
Target: 10-12% low confidence (60-75/613)

This represents a 50% reduction in human review workload while maintaining accuracy.
