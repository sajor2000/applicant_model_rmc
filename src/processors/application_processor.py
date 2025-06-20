"""Application processing module for the AI admissions system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from ..features.feature_engineer import FeatureEngineer
from ..features.essay_analyzer import EssayAnalyzer
from ..models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ApplicationProcessor:
    """Process medical school applications through the AI pipeline."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the processor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_loader = ModelLoader()
        self.feature_engineer = FeatureEngineer()
        self.essay_analyzer = EssayAnalyzer()
        
        # Load model
        if model_path:
            self.load_model(model_path)
        else:
            self.load_latest_model()
    
    def load_model(self, model_path: Path) -> None:
        """Load a specific model file."""
        model_data = self.model_loader.load_model(model_path)
        self.classifier = model_data['classifier']
        self.feature_cols = model_data['feature_cols']
        self.imputer = model_data['imputer']
        self.scaler = model_data['scaler']
        logger.info(f"Loaded model from {model_path}")
    
    def load_latest_model(self) -> None:
        """Load the latest available model."""
        model_data = self.model_loader.load_latest_model()
        if model_data:
            self.classifier = model_data['classifier']
            self.feature_cols = model_data['feature_cols']
            self.imputer = model_data['imputer']
            self.scaler = model_data['scaler']
            logger.info("Loaded latest model")
        else:
            logger.warning("No model found")
    
    def process_single(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single application.
        
        Args:
            application: Dictionary with application data
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([application])
            
            # Process essay if provided
            if 'essay_text' in application and application['essay_text']:
                essay_features = self.essay_analyzer.analyze_essay(
                    application['essay_text'],
                    application.get('amcas_id', 'unknown')
                )
                
                # Add essay features to dataframe
                for feature, value in essay_features.items():
                    df[feature] = value
            
            # Engineer features
            df_engineered = self.feature_engineer.engineer_features(df)
            
            # Prepare for model
            X = self._prepare_features(df_engineered)
            
            # Make prediction
            results = self.classifier.predict_quartiles(X)
            
            # Extract single result
            return {
                'success': True,
                'amcas_id': application.get('amcas_id'),
                'predicted_quartile': results['quartiles'][0],
                'confidence': float(results['confidences'][0]),
                'needs_review': bool(results['needs_review'][0]),
                'probabilities': {
                    'Q1': float(results['probabilities'][0, 3]),
                    'Q2': float(results['probabilities'][0, 2]),
                    'Q3': float(results['probabilities'][0, 1]),
                    'Q4': float(results['probabilities'][0, 0])
                },
                'top_features': self._extract_top_features(df_engineered)
            }
            
        except Exception as e:
            logger.error(f"Error processing application: {e}")
            return {
                'success': False,
                'error': str(e),
                'amcas_id': application.get('amcas_id')
            }
    
    def process_batch(self, applications: pd.DataFrame, 
                     progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """Process multiple applications in batch.
        
        Args:
            applications: DataFrame with application data
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with predictions and metadata
        """
        results = []
        total = len(applications)
        
        for idx, row in applications.iterrows():
            # Process single application
            result = self.process_single(row.to_dict())
            results.append(result)
            
            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original data
        if 'amcas_id' in applications.columns:
            results_df = results_df.merge(
                applications, 
                on='amcas_id', 
                how='left',
                suffixes=('', '_original')
            )
        
        return results_df
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        # Select only features used in training
        feature_data = df[self.feature_cols].copy()
        
        # Handle categorical variables
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1},
            'citizenship': {'US_Citizen': 0, 'Permanent_Resident': 1, 
                          'International': 2, 'Other': 3},
            'service_rating_categorical': {
                'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
                'Good': 2, 'Average': 1, 'Below Average': 0, 'Poor': -1
            }
        }
        
        for col, mapping in categorical_mappings.items():
            if col in feature_data.columns:
                feature_data[col] = feature_data[col].map(mapping).fillna(-2)
        
        # Convert remaining object columns
        for col in feature_data.select_dtypes(include=['object']).columns:
            feature_data[col] = pd.Categorical(feature_data[col]).codes
        
        # Impute and scale
        X_imputed = self.imputer.transform(feature_data)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def _extract_top_features(self, df: pd.DataFrame) -> List[str]:
        """Extract top contributing features for an application."""
        features = []
        
        # Key numeric features
        if 'service_rating_numerical' in df.columns:
            features.append(f"Service Rating: {df['service_rating_numerical'].iloc[0]}")
        
        if 'llm_overall_essay_score' in df.columns:
            score = df['llm_overall_essay_score'].iloc[0]
            features.append(f"Essay Score: {score:.0f}/100")
        
        if 'healthcare_total_hours' in df.columns:
            hours = df['healthcare_total_hours'].iloc[0]
            features.append(f"Clinical Hours: {hours:.0f}")
        
        # Engineered features
        if 'essay_service_alignment' in df.columns:
            alignment = df['essay_service_alignment'].iloc[0]
            if alignment > 0.8:
                features.append("Strong Essay-Service Alignment")
            elif alignment < 0.5:
                features.append("Weak Essay-Service Alignment")
        
        if 'profile_coherence' in df.columns:
            coherence = df['profile_coherence'].iloc[0]
            if coherence > 0.7:
                features.append("Highly Coherent Profile")
        
        return features[:5]