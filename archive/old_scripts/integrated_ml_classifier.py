"""
Integrated ML Classifier: Combining Structured Data + LLM Scores
===============================================================

This module implements the final classifier that combines:
1. Structured features from Excel files (GPA, MCAT, hours, etc.)
2. Numeric scores from LLM essay evaluation
3. Interaction features between the two

The output is a single tier prediction (1-4) with confidence scores.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple
import json


class IntegratedAdmissionsClassifier:
    """
    Final ML classifier that combines structured and LLM-derived features
    """
    
    def __init__(self):
        # Define all features used in the model
        self.structured_features = [
            'age', 'gpa_total', 'mcat_total', 'exp_hour_total',
            'exp_hour_research', 'exp_hour_clinical', 'exp_hour_volunteer_non_med',
            'service_rating_numerical', 'gpa_trend_total', 'gpa_trend_bcpm',
            'disadvantaged_ind', 'first_generation_ind', 'ses_disadvantaged',
            'num_dependents', 'comm_service_hours', 'healthcare_hours'
        ]
        
        self.engineered_features = [
            'research_intensity', 'clinical_intensity', 'experience_balance',
            'service_commitment', 'adversity_overcome'
        ]
        
        self.llm_features = [
            'llm_motivation_score', 'llm_clinical_understanding', 'llm_service_commitment',
            'llm_resilience_score', 'llm_academic_readiness', 'llm_interpersonal_skills',
            'llm_leadership_score', 'llm_ethical_maturity', 'llm_overall_normalized',
            'llm_confidence_score', 'llm_red_flag_count', 'llm_green_flag_count'
        ]
        
        self.interaction_features = [
            'experience_quality_score', 'service_impact_score', 'academic_potential_score',
            'leadership_effectiveness', 'clinical_depth_score'
        ]
        
        self.all_features = (
            self.structured_features + 
            self.engineered_features + 
            self.llm_features + 
            self.interaction_features
        )
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def engineer_structured_features(self, data: Dict) -> Dict:
        """
        Create engineered features from structured data
        """
        epsilon = 1e-6
        
        # Ensure numeric types
        for key in ['exp_hour_total', 'exp_hour_research', 'exp_hour_clinical', 
                   'exp_hour_volunteer_non_med', 'comm_service_hours']:
            data[key] = float(data.get(key, 0))
        
        engineered = {}
        
        # Research intensity: proportion of research in total experience
        engineered['research_intensity'] = (
            data['exp_hour_research'] / (data['exp_hour_total'] + epsilon)
        )
        
        # Clinical intensity: proportion of clinical work
        engineered['clinical_intensity'] = (
            (data['exp_hour_clinical'] + data.get('exp_hour_shadowing', 0)) / 
            (data['exp_hour_total'] + epsilon)
        )
        
        # Experience balance: research vs clinical ratio
        clinical_total = data['exp_hour_clinical'] + data.get('exp_hour_shadowing', 0)
        engineered['experience_balance'] = (
            data['exp_hour_research'] / (clinical_total + epsilon)
        )
        
        # Service commitment: service rating × log(hours)
        engineered['service_commitment'] = (
            data.get('service_rating_numerical', 0) * 
            np.log(data['comm_service_hours'] + 1)
        )
        
        # Adversity overcome: disadvantaged × GPA trend
        engineered['adversity_overcome'] = (
            int(data.get('disadvantaged_ind', 0)) * 
            data.get('gpa_trend_total', 0)
        )
        
        return engineered
    
    def create_interaction_features(self, structured: Dict, llm_scores: Dict) -> Dict:
        """
        Create features that capture interaction between structured and LLM scores
        """
        interactions = {}
        
        # Experience quality: hours × LLM understanding
        interactions['experience_quality_score'] = (
            structured['exp_hour_total'] * 
            llm_scores['llm_clinical_understanding'] / 10.0
        )
        
        # Service impact: service rating × LLM service score
        interactions['service_impact_score'] = (
            structured.get('service_rating_numerical', 0) * 
            llm_scores['llm_service_commitment'] / 10.0
        )
        
        # Academic potential: GPA × LLM academic readiness
        interactions['academic_potential_score'] = (
            structured.get('gpa_total', 0) * 
            llm_scores['llm_academic_readiness'] / 10.0
        )
        
        # Leadership effectiveness: hours in leadership × LLM leadership score
        leadership_hours = structured.get('exp_hour_leadership', 0)
        interactions['leadership_effectiveness'] = (
            leadership_hours * llm_scores['llm_leadership_score'] / 10.0
        )
        
        # Clinical depth: clinical hours × LLM clinical understanding
        clinical_hours = (
            structured.get('exp_hour_clinical', 0) + 
            structured.get('exp_hour_shadowing', 0)
        )
        interactions['clinical_depth_score'] = (
            clinical_hours * llm_scores['llm_clinical_understanding'] / 10.0
        )
        
        return interactions
    
    def prepare_feature_vector(self, 
                             structured_data: Dict, 
                             llm_scores: Dict) -> np.ndarray:
        """
        Combine all features into a single vector for ML model
        """
        # Engineer structured features
        engineered = self.engineer_structured_features(structured_data)
        
        # Normalize LLM overall score to 0-10 scale
        llm_scores['llm_overall_normalized'] = llm_scores.get('llm_overall_score', 50) / 10.0
        
        # Create interaction features
        interactions = self.create_interaction_features(structured_data, llm_scores)
        
        # Combine all features
        feature_vector = []
        
        # Add structured features
        for feature in self.structured_features:
            value = structured_data.get(feature, 0)
            # Convert boolean indicators to 0/1
            if feature.endswith('_ind'):
                value = 1 if value in [True, 'Yes', 'yes', 1] else 0
            feature_vector.append(float(value))
        
        # Add engineered features
        for feature in self.engineered_features:
            feature_vector.append(engineered[feature])
        
        # Add LLM features
        for feature in self.llm_features:
            feature_vector.append(float(llm_scores.get(feature, 0)))
        
        # Add interaction features
        for feature in self.interaction_features:
            feature_vector.append(interactions[feature])
        
        return np.array(feature_vector)
    
    def predict_tier(self, 
                    structured_data: Dict, 
                    llm_scores: Dict) -> Dict:
        """
        Make final tier prediction combining all features
        """
        # Prepare feature vector
        X = self.prepare_feature_vector(structured_data, llm_scores).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        tier_prediction = self.model.predict(X_scaled)[0]
        tier_probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get feature contributions (for explainability)
        feature_contributions = self.get_feature_contributions(X[0])
        
        # Determine which component (structured vs LLM) had more influence
        structured_importance = sum(
            feature_contributions[f] for f in self.structured_features + self.engineered_features
        )
        llm_importance = sum(
            feature_contributions[f] for f in self.llm_features
        )
        interaction_importance = sum(
            feature_contributions[f] for f in self.interaction_features
        )
        
        total_importance = structured_importance + llm_importance + interaction_importance
        
        return {
            'final_tier': int(tier_prediction),
            'tier_name': self.get_tier_name(tier_prediction),
            'tier_probabilities': {
                'very_unlikely': float(tier_probabilities[0]),
                'potential_review': float(tier_probabilities[1]),
                'probable_interview': float(tier_probabilities[2]),
                'very_likely_interview': float(tier_probabilities[3])
            },
            'confidence': float(tier_probabilities.max()),
            'interview_recommendation': 'YES' if tier_prediction >= 2 else 'NO',
            
            # Component contributions
            'component_weights': {
                'structured_data': structured_importance / total_importance,
                'llm_evaluation': llm_importance / total_importance,
                'interaction_effects': interaction_importance / total_importance
            },
            
            # Top contributing features
            'top_positive_factors': self.get_top_factors(feature_contributions, positive=True),
            'top_negative_factors': self.get_top_factors(feature_contributions, positive=False)
        }
    
    def get_tier_name(self, tier: int) -> str:
        """Convert tier number to name"""
        tier_names = {
            0: 'Very Unlikely',
            1: 'Potential Review',
            2: 'Probable Interview',
            3: 'Very Likely Interview'
        }
        return tier_names.get(tier, 'Unknown')
    
    def get_feature_contributions(self, feature_vector: np.ndarray) -> Dict:
        """
        Calculate feature contributions to the prediction
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {name: 0 for name in self.all_features}
        
        # Get feature importances from model
        importances = self.model.feature_importances_
        
        # Weight by actual feature values
        contributions = {}
        for i, (name, value) in enumerate(zip(self.all_features, feature_vector)):
            contributions[name] = importances[i] * value
        
        return contributions
    
    def get_top_factors(self, contributions: Dict, positive: bool = True, top_n: int = 5) -> List:
        """
        Get top contributing factors
        """
        # Sort by contribution
        sorted_factors = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=positive
        )
        
        # Get top N
        top_factors = []
        for factor, contribution in sorted_factors[:top_n]:
            if (positive and contribution > 0) or (not positive and contribution < 0):
                # Make factor names human-readable
                readable_name = factor.replace('_', ' ').title()
                readable_name = readable_name.replace('Llm ', 'Essay: ')
                readable_name = readable_name.replace('Gpa', 'GPA')
                readable_name = readable_name.replace('Mcat', 'MCAT')
                
                top_factors.append({
                    'factor': readable_name,
                    'impact': abs(contribution)
                })
        
        return top_factors
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Train the integrated model
        """
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Train model
        self.model.fit(X_scaled, y_train)
        
        # Store feature importance
        self.feature_importance = dict(zip(self.all_features, self.model.feature_importances_))
        
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.all_features,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.all_features = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']


# Example usage
def evaluate_with_integrated_model(applicant_data: Dict) -> Dict:
    """
    Complete evaluation using integrated model
    """
    # Initialize classifier
    classifier = IntegratedAdmissionsClassifier()
    classifier.load_model('models/integrated_model_2022_2023.pkl')
    
    # Separate structured data and LLM scores
    structured_data = {
        'age': applicant_data['Age'],
        'gpa_total': applicant_data['Total_GPA'],
        'mcat_total': applicant_data['MCAT_Total_Score'],
        'exp_hour_total': applicant_data['Exp_Hour_Total'],
        'exp_hour_research': applicant_data['Exp_Hour_Research'],
        'exp_hour_clinical': applicant_data['Exp_Hour_Volunteer_Med'],
        'service_rating_numerical': applicant_data['Service Rating (Numerical)'],
        # ... other structured fields
    }
    
    # LLM scores (these would come from Azure OpenAI)
    llm_scores = {
        'llm_motivation_score': 8.5,
        'llm_clinical_understanding': 7.0,
        'llm_service_commitment': 8.0,
        'llm_resilience_score': 9.0,
        'llm_academic_readiness': 7.5,
        'llm_interpersonal_skills': 8.0,
        'llm_leadership_score': 6.5,
        'llm_ethical_maturity': 7.0,
        'llm_overall_score': 76.5,
        'llm_confidence_score': 0.85,
        'llm_red_flag_count': 0,
        'llm_green_flag_count': 3
    }
    
    # Get prediction
    result = classifier.predict_tier(structured_data, llm_scores)
    
    return result


if __name__ == "__main__":
    # Example evaluation
    sample_result = evaluate_with_integrated_model({
        'Age': 24,
        'Total_GPA': 3.7,
        'MCAT_Total_Score': 512,
        'Exp_Hour_Total': 2000,
        'Exp_Hour_Research': 800,
        'Exp_Hour_Volunteer_Med': 600,
        'Service Rating (Numerical)': 3
    })
    
    print(json.dumps(sample_result, indent=2))