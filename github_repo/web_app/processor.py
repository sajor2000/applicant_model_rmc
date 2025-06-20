"""
Application Processor Module
Handles the AI processing of medical school applications
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


class ApplicationProcessor:
    """Process applications through the trained AI model."""
    
    def __init__(self):
        """Initialize the processor with model and API connections."""
        self.model = None
        self.feature_cols = None
        self.imputer = None
        self.scaler = None
        self.azure_client = None
        
        # Load the latest model
        self._load_model()
        
        # Initialize Azure OpenAI
        self._init_azure_openai()
    
    def _load_model(self):
        """Load the latest trained model."""
        model_path = Path("models")
        if model_path.exists():
            # Find the latest model file
            model_files = sorted(model_path.glob("*cascade*.pkl"))
            if model_files:
                latest_model = model_files[-1]
                print(f"Loading model: {latest_model}")
                
                model_data = joblib.load(latest_model)
                self.model = model_data['optimizer']
                self.feature_cols = model_data['feature_cols']
                self.imputer = model_data['imputer']
                self.scaler = model_data['scaler']
                
                return True
        
        print("Warning: No model found. Please ensure model file exists.")
        return False
    
    def _init_azure_openai(self):
        """Initialize Azure OpenAI client."""
        try:
            self.azure_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        except Exception as e:
            print(f"Warning: Could not initialize Azure OpenAI: {e}")
            self.azure_client = None
    
    def process_essays(self, essay_text, amcas_id):
        """Process essays through GPT-4o to extract features."""
        if not self.azure_client:
            # Return default values if no API connection
            return self._get_default_essay_features()
        
        prompts = {
            'overall_essay_score': "Rate the overall quality of this medical school application essay on a scale of 0-100, considering writing quality, coherence, impact, and authenticity.",
            'motivation_authenticity': "Rate from 0-100 how genuine and personal the applicant's motivations for medicine appear to be.",
            'clinical_insight': "Rate from 0-100 the depth of clinical understanding and realistic view of medicine demonstrated.",
            'leadership_impact': "Rate from 0-100 the evidence of leadership with tangible outcomes and initiative shown.",
            'service_genuineness': "Rate from 0-100 the applicant's genuine commitment to serving others versus resume building.",
            'intellectual_curiosity': "Rate from 0-100 the evidence of intellectual engagement, research interest, and growth mindset.",
            'maturity_score': "Rate from 0-100 the emotional maturity, self-awareness, and professional readiness shown.",
            'communication_score': "Rate from 0-100 the effectiveness of written communication, clarity, and persuasiveness.",
            'diversity_contribution': "Rate from 0-100 the unique perspectives and contributions to medical school diversity offered.",
            'resilience_score': "Rate from 0-100 the evidence of resilience, perseverance, and growth from challenges.",
            'ethical_reasoning': "Rate from 0-100 the understanding of medical ethics, integrity, and professional values shown.",
            'red_flags': "Count the number of concerning elements (professionalism lapses, unrealistic views, concerning behaviors).",
            'green_flags': "Count the number of exceptional elements (outstanding achievements, exceptional insights, unique strengths)."
        }
        
        essay_features = {}
        
        try:
            for feature, prompt in prompts.items():
                response = self.azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert medical school admissions evaluator. Provide only a numeric score or count as requested."},
                        {"role": "user", "content": f"{prompt}\n\nEssay:\n{essay_text[:4000]}"}  # Limit text length
                    ],
                    temperature=0.15,
                    top_p=0.9,
                    max_tokens=10
                )
                
                try:
                    if 'flags' in feature:
                        essay_features[f'llm_{feature}_count'] = int(response.choices[0].message.content.strip())
                    else:
                        essay_features[f'llm_{feature}'] = float(response.choices[0].message.content.strip())
                except:
                    essay_features[f'llm_{feature}'] = 0
                    
        except Exception as e:
            print(f"Error processing essays for {amcas_id}: {e}")
            return self._get_default_essay_features()
        
        return essay_features
    
    def _get_default_essay_features(self):
        """Return default essay features when API is unavailable."""
        return {
            'llm_overall_essay_score': 70.0,
            'llm_motivation_authenticity': 70.0,
            'llm_clinical_insight': 70.0,
            'llm_leadership_impact': 70.0,
            'llm_service_genuineness': 70.0,
            'llm_intellectual_curiosity': 70.0,
            'llm_maturity_score': 70.0,
            'llm_communication_score': 70.0,
            'llm_diversity_contribution': 70.0,
            'llm_resilience_score': 70.0,
            'llm_ethical_reasoning': 70.0,
            'llm_red_flag_count': 0,
            'llm_green_flag_count': 1
        }
    
    def engineer_features(self, df):
        """Create engineered features matching the training pipeline."""
        df = df.copy()
        
        # Essay-service alignment
        if 'llm_overall_essay_score' in df.columns and 'service_rating_numerical' in df.columns:
            essay_norm = df['llm_overall_essay_score'] / 100
            service_norm = (df['service_rating_numerical'] - 1) / 3
            df['essay_service_alignment'] = 1 - abs(essay_norm - service_norm)
            df['service_essay_product'] = df['service_rating_numerical'] * df['llm_overall_essay_score'] / 25
        
        # Flag features
        if 'llm_red_flag_count' in df.columns and 'llm_green_flag_count' in df.columns:
            df['flag_balance'] = df['llm_green_flag_count'] - df['llm_red_flag_count']
            df['flag_ratio'] = df['llm_green_flag_count'] / (df['llm_red_flag_count'] + 1)
        
        # Service × Clinical interaction
        if 'service_rating_numerical' in df.columns and 'healthcare_total_hours' in df.columns:
            df['service_clinical_log'] = df['service_rating_numerical'] * np.log1p(df['healthcare_total_hours'])
        
        # Experience features
        exp_cols = ['exp_hour_research', 'exp_hour_volunteer_med', 'exp_hour_volunteer_non_med']
        available_exp = [col for col in exp_cols if col in df.columns]
        if len(available_exp) > 1:
            df['experience_diversity'] = (df[available_exp] > 50).sum(axis=1)
            exp_mean = df[available_exp].mean(axis=1)
            exp_std = df[available_exp].std(axis=1)
            df['experience_consistency'] = 1 / (1 + exp_std / (exp_mean + 1))
        
        # Add any other engineered features from training
        
        return df
    
    def process_single_application(self, application_data):
        """Process a single application through the model."""
        if not self.model:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Convert to dataframe if needed
            if isinstance(application_data, dict):
                df = pd.DataFrame([application_data])
            else:
                df = application_data.copy()
            
            # Process essays if text provided
            if 'essay_text' in df.columns and df['essay_text'].iloc[0]:
                essay_features = self.process_essays(
                    df['essay_text'].iloc[0],
                    df['amcas_id'].iloc[0]
                )
                
                # Add essay features to dataframe
                for feature, value in essay_features.items():
                    df[feature] = value
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Prepare features for model
            X = self._prepare_features(df)
            
            # Make prediction
            predictions, probabilities, confidences = self.model.predict_with_enhanced_confidence(X)
            
            # Convert to quartile
            quartile_map = {0: 'Q4', 1: 'Q3', 2: 'Q2', 3: 'Q1'}
            predicted_quartile = quartile_map[predictions[0]]
            
            # Determine if needs review
            needs_review = confidences[0] < 80 or predicted_quartile in ['Q2', 'Q3']
            
            result = {
                'success': True,
                'amcas_id': df['amcas_id'].iloc[0],
                'predicted_quartile': predicted_quartile,
                'confidence': float(confidences[0]),
                'needs_review': needs_review,
                'reject_probability': float(probabilities[0, 0]),
                'waitlist_probability': float(probabilities[0, 1]),
                'interview_probability': float(probabilities[0, 2]),
                'accept_probability': float(probabilities[0, 3]),
                'top_features': self._get_top_features(df)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, df, progress_callback=None):
        """Process multiple applications in batch."""
        results = []
        total = len(df)
        
        for idx, row in df.iterrows():
            # Process single application
            result = self.process_single_application(row.to_dict())
            results.append(result)
            
            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total)
        
        return pd.DataFrame(results)
    
    def _prepare_features(self, df):
        """Prepare features for model input."""
        # Get only the features used in training
        feature_data = df[self.feature_cols].copy()
        
        # Handle categorical variables
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1},
            'citizenship': {'US_Citizen': 0, 'Permanent_Resident': 1, 'International': 2, 'Other': 3}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in feature_data.columns:
                feature_data[col] = feature_data[col].map(mapping).fillna(-2)
        
        # Convert any remaining categoricals
        for col in feature_data.select_dtypes(include=['object']).columns:
            feature_data[col] = pd.Categorical(feature_data[col]).codes
        
        # Impute and scale
        X_imputed = self.imputer.transform(feature_data)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def _get_top_features(self, df):
        """Extract top contributing features for this application."""
        top_features = []
        
        # Check key features
        if 'service_rating_numerical' in df.columns:
            top_features.append(f"Service Rating: {df['service_rating_numerical'].iloc[0]}")
        
        if 'llm_overall_essay_score' in df.columns:
            top_features.append(f"Essay Score: {df['llm_overall_essay_score'].iloc[0]:.0f}")
        
        if 'healthcare_total_hours' in df.columns:
            top_features.append(f"Clinical Hours: {df['healthcare_total_hours'].iloc[0]:.0f}")
        
        if 'essay_service_alignment' in df.columns:
            alignment = df['essay_service_alignment'].iloc[0]
            if alignment > 0.8:
                top_features.append("Strong Essay-Service Alignment")
            elif alignment < 0.5:
                top_features.append("Weak Essay-Service Alignment")
        
        return top_features[:5]  # Return top 5 features