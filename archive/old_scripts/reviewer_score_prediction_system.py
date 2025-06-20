"""
Reviewer Score Prediction System
================================

Core Objective: Predict 'Application Review Score' (0-25) with high fidelity
using both structured data and LLM-evaluated unstructured content.

The Application Review Score from trusted reviewers is our ground truth.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


class ReviewerScorePredictionSystem:
    """
    Predicts the Application Review Score (0-25) using all available data.
    This score is our ground truth from trusted reviewers.
    """
    
    def __init__(self):
        # Target variable
        self.target = 'Application Review Score'  # 0-25 scale from reviewers
        
        # We'll use regression instead of classification for precise score prediction
        self.model = None
        self.scaler = StandardScaler()
        
        # Track feature importance for understanding what drives reviewer scores
        self.feature_importance = {}
        
        # Define features that correlate with reviewer decisions
        self.structured_features = [
            # Academic metrics (likely weighted heavily by reviewers)
            'gpa_total', 'gpa_science', 'mcat_total', 'mcat_cpbs', 'mcat_cars',
            'gpa_trend_total', 'gpa_trend_bcpm',
            
            # Experience quantity and diversity
            'exp_hour_total', 'exp_hour_research', 'exp_hour_clinical',
            'exp_hour_volunteer_med', 'exp_hour_volunteer_non_med',
            'exp_hour_shadowing', 'exp_hour_leadership',
            
            # Service and commitment indicators
            'service_rating_numerical',  # This already exists in data
            'comm_service_total_hours', 'healthcare_total_hours',
            
            # Contextual factors reviewers consider
            'disadvantaged_ind', 'first_generation_ind', 'age',
            'num_activities', 'num_meaningful_experiences'
        ]
        
        # LLM features designed to capture what reviewers value in essays
        self.llm_features = [
            # Essay quality scores that reviewers would notice
            'llm_narrative_coherence',      # How well the story flows
            'llm_motivation_authenticity',   # Genuine vs generic motivation
            'llm_reflection_depth',          # Surface vs deep reflection
            'llm_growth_demonstrated',       # Evidence of personal development
            'llm_unique_perspective',        # What makes them stand out
            'llm_clinical_insight',          # Understanding of medicine
            'llm_service_genuineness',       # Authentic commitment to service
            'llm_leadership_impact',         # Actual impact vs titles
            'llm_communication_quality',     # Writing clarity and engagement
            'llm_maturity_score',           # Professional and emotional maturity
            
            # Red/green flags that influence reviewer scores
            'llm_red_flag_severity',        # 0-10: How concerning are issues
            'llm_green_flag_strength',      # 0-10: How impressive are strengths
            
            # Overall essay assessment
            'llm_essay_overall_score'       # 0-100: Overall essay quality
        ]
        
    def analyze_reviewer_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze what factors correlate most with reviewer scores
        """
        correlations = {}
        
        # Check correlation of each feature with Application Review Score
        for feature in df.columns:
            if feature != self.target and pd.api.types.is_numeric_dtype(df[feature]):
                corr = df[feature].corr(df[self.target])
                if not pd.isna(corr):
                    correlations[feature] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("Top 15 Features Correlated with Reviewer Scores:")
        print("-" * 50)
        for feature, corr in sorted_corr[:15]:
            print(f"{feature:40s}: {corr:+.3f}")
        
        return dict(sorted_corr)
    
    def create_llm_prompt_for_reviewer_prediction(self, essays: Dict) -> str:
        """
        Create a prompt specifically designed to predict reviewer scores
        """
        prompt = """You are an experienced medical school application reviewer. Your task is to evaluate these application materials and predict how other reviewers would score this applicant.

IMPORTANT: Focus on factors that typically influence reviewer scores:
1. Narrative quality and coherence
2. Depth of clinical understanding
3. Evidence of personal growth
4. Authenticity of motivation
5. Communication effectiveness
6. Maturity and professionalism
7. Unique contributions to medicine
8. Red flags or concerns

ESSAYS TO EVALUATE:

Personal Statement:
{personal_statement}

Secondary Essays:
{secondary_essays}

Experience Descriptions:
{experiences}

SCORING TASK:
Based on these materials, provide numeric scores (0-10 unless specified) for:

1. llm_narrative_coherence: How well does their story hold together?
2. llm_motivation_authenticity: How genuine vs generic is their motivation?
3. llm_reflection_depth: How deeply do they reflect on experiences?
4. llm_growth_demonstrated: Evidence of personal development?
5. llm_unique_perspective: What unique value do they bring?
6. llm_clinical_insight: Understanding of healthcare realities?
7. llm_service_genuineness: Authentic commitment to service?
8. llm_leadership_impact: Actual impact created?
9. llm_communication_quality: Writing clarity and engagement?
10. llm_maturity_score: Professional and emotional maturity?
11. llm_red_flag_severity: How concerning are any issues? (0=none, 10=very)
12. llm_green_flag_strength: How impressive are strengths? (0=none, 10=exceptional)
13. llm_essay_overall_score: Overall essay quality (0-100)

CRITICAL: These scores should predict how application reviewers would rate this candidate on a 0-25 scale. Consider what makes reviewers give high scores (20-25) vs low scores (0-15).

Return as JSON with all numeric scores."""

        return prompt.format(
            personal_statement=essays.get('personal_statement', '')[:3000],
            secondary_essays=self._format_secondary_essays(essays.get('secondary_essays', {})),
            experiences=self._format_experiences(essays.get('experiences', []))
        )
    
    def _format_secondary_essays(self, essays: Dict) -> str:
        """Format secondary essays for prompt"""
        formatted = ""
        for key, essay in list(essays.items())[:4]:  # Limit to 4
            formatted += f"\n{key}:\n{essay[:500]}\n"
        return formatted
    
    def _format_experiences(self, experiences: List) -> str:
        """Format experience descriptions for prompt"""
        formatted = ""
        for i, exp in enumerate(experiences[:3]):  # Limit to 3
            formatted += f"\nExperience {i+1}:\n"
            if 'description' in exp:
                formatted += f"{exp['description'][:300]}\n"
            if 'meaningful_reflection' in exp:
                formatted += f"Why meaningful: {exp['meaningful_reflection'][:300]}\n"
        return formatted
    
    def engineer_features(self, structured_data: Dict, llm_scores: Dict) -> Dict:
        """
        Engineer features specifically designed to predict reviewer scores
        """
        features = {}
        
        # Direct features
        features.update(structured_data)
        features.update(llm_scores)
        
        # Interaction features that might influence reviewers
        
        # 1. Academic trajectory: GPA trend × overall GPA
        features['academic_trajectory'] = (
            structured_data.get('gpa_total', 0) * 
            structured_data.get('gpa_trend_total', 0)
        )
        
        # 2. Research productivity: Hours × quality
        features['research_productivity'] = (
            structured_data.get('exp_hour_research', 0) * 
            llm_scores.get('llm_reflection_depth', 5) / 10
        )
        
        # 3. Clinical preparedness: Hours × understanding
        features['clinical_preparedness'] = (
            (structured_data.get('exp_hour_clinical', 0) + 
             structured_data.get('exp_hour_shadowing', 0)) *
            llm_scores.get('llm_clinical_insight', 5) / 10
        )
        
        # 4. Service authenticity: Rating × genuineness
        features['service_authenticity'] = (
            structured_data.get('service_rating_numerical', 0) *
            llm_scores.get('llm_service_genuineness', 5) / 10
        )
        
        # 5. Overall package strength: Academic × Essay × Experience
        features['package_strength'] = (
            (structured_data.get('gpa_total', 0) / 4.0) *
            (structured_data.get('mcat_total', 500) / 528) *
            (llm_scores.get('llm_essay_overall_score', 50) / 100) *
            100  # Scale to 0-100
        )
        
        # 6. Red flag adjustment (reviewers penalize heavily)
        features['red_flag_penalty'] = (
            llm_scores.get('llm_red_flag_severity', 0) * -2  # Negative impact
        )
        
        # 7. Exceptional bonus (reviewers reward standouts)
        features['exceptional_bonus'] = (
            llm_scores.get('llm_green_flag_strength', 0) *
            llm_scores.get('llm_unique_perspective', 0) / 10
        )
        
        return features
    
    def train_reviewer_prediction_model(self, 
                                      X_train: pd.DataFrame, 
                                      y_train: pd.Series,
                                      model_type: str = 'xgboost'):
        """
        Train model to predict Application Review Scores
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42
            )
        else:  # gradient boosting
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_scaled, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X_train.columns, 
                self.model.feature_importances_
            ))
        
        # Cross-validation to assess performance
        cv_scores = cross_val_score(
            self.model, X_scaled, y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        
        print(f"\nModel Performance (5-fold CV):")
        print(f"Mean Absolute Error: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        print(f"As % of scale: {(-cv_scores.mean()/25)*100:.1f}%")
        
        return self.model
    
    def predict_reviewer_score(self, 
                              structured_data: Dict, 
                              llm_scores: Dict) -> Dict:
        """
        Predict the Application Review Score (0-25)
        """
        # Engineer features
        all_features = self.engineer_features(structured_data, llm_scores)
        
        # Create feature vector in correct order
        feature_vector = pd.DataFrame([all_features])[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(feature_vector)
        
        # Predict score
        predicted_score = self.model.predict(X_scaled)[0]
        
        # Ensure score is in valid range
        predicted_score = np.clip(predicted_score, 0, 25)
        
        # Convert to tier (matching original logic)
        if predicted_score <= 14:
            tier = 1  # Very Unlikely
        elif predicted_score <= 18:
            tier = 2  # Potential Review
        elif predicted_score <= 22:
            tier = 3  # Probable Interview
        else:
            tier = 4  # Very Likely Interview
        
        # Get prediction interval (if model supports it)
        if hasattr(self.model, 'predict_proba'):
            # For classification models converted to regression
            confidence_interval = self._estimate_confidence_interval(X_scaled)
        else:
            # Simple estimate based on training error
            confidence_interval = (predicted_score - 2, predicted_score + 2)
        
        return {
            'predicted_review_score': float(predicted_score),
            'confidence_interval': confidence_interval,
            'predicted_tier': tier,
            'tier_name': self._get_tier_name(tier),
            'interview_recommendation': 'YES' if tier >= 3 else 'NO',
            
            # Feature contributions
            'top_positive_factors': self._get_top_factors(all_features, positive=True),
            'top_negative_factors': self._get_top_factors(all_features, positive=False),
            
            # Component analysis
            'score_breakdown': {
                'academic_contribution': self._calculate_component_contribution('academic', all_features),
                'experience_contribution': self._calculate_component_contribution('experience', all_features),
                'essay_contribution': self._calculate_component_contribution('essay', all_features),
                'interaction_contribution': self._calculate_component_contribution('interaction', all_features)
            }
        }
    
    def _get_tier_name(self, tier: int) -> str:
        """Convert tier to name"""
        return {
            1: 'Very Unlikely',
            2: 'Potential Review',
            3: 'Probable Interview',
            4: 'Very Likely Interview'
        }.get(tier, 'Unknown')
    
    def _estimate_confidence_interval(self, X_scaled: np.ndarray, confidence: float = 0.8):
        """Estimate confidence interval for prediction"""
        # This is a simplified approach
        # In practice, you might use quantile regression or bootstrapping
        prediction = self.model.predict(X_scaled)[0]
        # Rough estimate: ±8% of the scale
        margin = 25 * 0.08
        return (max(0, prediction - margin), min(25, prediction + margin))
    
    def _get_top_factors(self, features: Dict, positive: bool = True, n: int = 5) -> List[Dict]:
        """Get top contributing factors based on feature importance"""
        if not self.feature_importance:
            return []
        
        # Calculate contribution of each feature
        contributions = {}
        for feature, value in features.items():
            if feature in self.feature_importance:
                contribution = self.feature_importance[feature] * value
                contributions[feature] = contribution
        
        # Sort and get top N
        sorted_contrib = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=positive
        )
        
        top_factors = []
        for feature, contrib in sorted_contrib[:n]:
            if (positive and contrib > 0) or (not positive and contrib < 0):
                top_factors.append({
                    'factor': self._humanize_feature_name(feature),
                    'impact': abs(contrib)
                })
        
        return top_factors
    
    def _humanize_feature_name(self, feature: str) -> str:
        """Convert feature name to human-readable format"""
        replacements = {
            'llm_': 'Essay: ',
            'exp_hour_': 'Hours: ',
            'gpa_': 'GPA ',
            'mcat_': 'MCAT ',
            '_': ' '
        }
        
        name = feature
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name.title()
    
    def _calculate_component_contribution(self, component: str, features: Dict) -> float:
        """Calculate how much each component contributes to the prediction"""
        component_features = {
            'academic': ['gpa_', 'mcat_', 'academic_'],
            'experience': ['exp_hour_', 'service_', 'clinical_', 'research_'],
            'essay': ['llm_'],
            'interaction': ['_trajectory', '_productivity', '_preparedness', '_authenticity', 'package_', 'exceptional_']
        }
        
        total_contribution = 0
        for feature, value in features.items():
            if feature in self.feature_importance:
                for pattern in component_features.get(component, []):
                    if pattern in feature:
                        total_contribution += self.feature_importance[feature] * abs(value)
                        break
        
        return total_contribution
    
    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate how well we predict reviewer scores
        """
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        
        # Ensure predictions are in valid range
        y_pred = np.clip(y_pred, 0, 25)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Check tier accuracy (more interpretable)
        y_test_tiers = pd.cut(y_test, bins=[-1, 14, 18, 22, 25], labels=[1, 2, 3, 4])
        y_pred_tiers = pd.cut(y_pred, bins=[-1, 14, 18, 22, 25], labels=[1, 2, 3, 4])
        tier_accuracy = (y_test_tiers == y_pred_tiers).mean()
        
        # Adjacent tier accuracy (within 1 tier)
        tier_diff = abs(y_test_tiers.astype(int) - y_pred_tiers.astype(int))
        adjacent_accuracy = (tier_diff <= 1).mean()
        
        results = {
            'mean_absolute_error': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mae_as_percent': (mae / 25) * 100,
            'tier_exact_accuracy': tier_accuracy,
            'tier_adjacent_accuracy': adjacent_accuracy,
            'correlation': np.corrcoef(y_test, y_pred)[0, 1]
        }
        
        print("\nModel Performance on Test Set:")
        print(f"MAE: {mae:.2f} points (out of 25)")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")
        print(f"Correlation: {results['correlation']:.3f}")
        print(f"Exact Tier Accuracy: {tier_accuracy:.1%}")
        print(f"Adjacent Tier Accuracy: {adjacent_accuracy:.1%}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, 25], [0, 25], 'r--', lw=2)
        plt.xlabel('Actual Reviewer Score')
        plt.ylabel('Predicted Reviewer Score')
        plt.title('Reviewer Score Prediction Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reviewer_score_predictions.png')
        
        return results


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_excel('data/2022 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx')
    
    # Initialize system
    system = ReviewerScorePredictionSystem()
    
    # Analyze what correlates with reviewer scores
    correlations = system.analyze_reviewer_patterns(df)
    
    # Example prediction
    structured_data = {
        'gpa_total': 3.7,
        'mcat_total': 512,
        'exp_hour_research': 800,
        'exp_hour_clinical': 600,
        'service_rating_numerical': 3
    }
    
    llm_scores = {
        'llm_narrative_coherence': 8.0,
        'llm_motivation_authenticity': 7.5,
        'llm_reflection_depth': 8.5,
        'llm_clinical_insight': 7.0,
        'llm_essay_overall_score': 75
    }
    
    prediction = system.predict_reviewer_score(structured_data, llm_scores)
    print(f"\nPredicted Reviewer Score: {prediction['predicted_review_score']:.1f}/25")