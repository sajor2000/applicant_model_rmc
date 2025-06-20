"""Tests for the application processor module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from github_repo.src.processors.application_processor import ApplicationProcessor


class TestApplicationProcessor:
    """Test suite for ApplicationProcessor."""
    
    @pytest.fixture
    def sample_application(self):
        """Sample application data for testing."""
        return {
            'amcas_id': 'TEST001',
            'service_rating_numerical': 4,
            'healthcare_total_hours': 1000,
            'exp_hour_research': 500,
            'exp_hour_volunteer_med': 300,
            'age': 24,
            'gender': 'Female',
            'citizenship': 'US_Citizen',
            'first_generation_ind': 0,
            'essay_text': 'Sample essay about medical aspirations...'
        }
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor with stubbed dependencies."""
        with patch('github_repo.src.processors.application_processor.ModelLoader') as mock_loader:
            with patch('github_repo.src.processors.application_processor.EssayAnalyzer') as mock_essay:
                with patch('github_repo.src.processors.application_processor.FeatureEngineer') as mock_feature:
                    processor = ApplicationProcessor()
                    
                    # Mock the classifier
                    processor.classifier = Mock()
                    processor.classifier.predict_quartiles.return_value = {
                        'quartiles': ['Q1'],
                        'confidences': np.array([85.0]),
                        'needs_review': np.array([False]),
                        'probabilities': np.array([[0.1, 0.1, 0.2, 0.6]])
                    }
                    
                    # Mock feature columns
                    processor.feature_cols = ['service_rating_numerical', 
                                            'healthcare_total_hours']
                    
                    # Mock preprocessors
                    processor.imputer = Mock()
                    processor.imputer.transform.return_value = np.array([[4, 1000]])
                    
                    processor.scaler = Mock()
                    processor.scaler.transform.return_value = np.array([[0.8, 0.7]])
                    
                    return processor
    
    def test_process_single_success(self, mock_processor, sample_application):
        """Test successful processing of single application."""
        result = mock_processor.process_single(sample_application)
        
        assert result['success'] is True
        assert result['amcas_id'] == 'TEST001'
        assert result['predicted_quartile'] == 'Q1'
        assert result['confidence'] == 85.0
        assert result['needs_review'] is False
        assert 'probabilities' in result
        assert result['probabilities']['Q1'] == 0.6
    
    def test_process_single_missing_essay(self, mock_processor, sample_application):
        """Test processing without essay text."""
        del sample_application['essay_text']
        
        result = mock_processor.process_single(sample_application)
        
        assert result['success'] is True
        assert result['predicted_quartile'] == 'Q1'
    
    def test_process_single_error_handling(self, mock_processor, sample_application):
        """Test error handling in single processing."""
        # Make the classifier raise an exception
        mock_processor.classifier.predict_quartiles.side_effect = Exception("Model error")
        
        result = mock_processor.process_single(sample_application)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['amcas_id'] == 'TEST001'
    
    def test_process_batch(self, mock_processor, sample_application):
        """Test batch processing of applications."""
        # Create batch dataframe
        batch_data = pd.DataFrame([sample_application] * 3)
        batch_data['amcas_id'] = ['TEST001', 'TEST002', 'TEST003']
        
        # Process batch
        results = mock_processor.process_batch(batch_data)
        
        assert len(results) == 3
        assert all(results['success'])
        assert list(results['amcas_id']) == ['TEST001', 'TEST002', 'TEST003']
    
    def test_process_batch_with_progress(self, mock_processor, sample_application):
        """Test batch processing with progress callback."""
        batch_data = pd.DataFrame([sample_application] * 2)
        
        # Track progress calls
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Process with callback
        results = mock_processor.process_batch(batch_data, progress_callback)
        
        assert len(results) == 2
        assert progress_calls == [(1, 2), (2, 2)]
    
    def test_prepare_features_categorical_handling(self, mock_processor):
        """Test categorical variable handling in feature preparation."""
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Other', 'Unknown'],
            'citizenship': ['US_Citizen', 'International', 'Permanent_Resident', 'Other'],
            'service_rating_numerical': [3, 4, 2, 3],
            'healthcare_total_hours': [1000, 800, 600, 1200]
        })
        
        # Set feature columns
        mock_processor.feature_cols = ['gender', 'citizenship', 
                                       'service_rating_numerical', 
                                       'healthcare_total_hours']
        
        # Prepare features
        X = mock_processor._prepare_features(df)
        
        # Check shape
        assert X.shape == (4, 4)
        
        # Verify imputer and scaler were called
        mock_processor.imputer.transform.assert_called_once()
        mock_processor.scaler.transform.assert_called_once()
    
    def test_extract_top_features(self, mock_processor):
        """Test extraction of top features for display."""
        df = pd.DataFrame({
            'service_rating_numerical': [4],
            'llm_overall_essay_score': [85],
            'healthcare_total_hours': [1200],
            'essay_service_alignment': [0.9],
            'profile_coherence': [0.8]
        })
        
        features = mock_processor._extract_top_features(df)
        
        assert len(features) <= 5
        assert "Service Rating: 4" in features
        assert "Essay Score: 85/100" in features
        assert "Clinical Hours: 1200" in features
        assert "Strong Essay-Service Alignment" in features
        assert "Highly Coherent Profile" in features


class TestProcessorIntegration:
    """Integration tests requiring actual model files."""
    
    @pytest.mark.skipif(not Path("models").exists(), 
                       reason="Model files not available")
    def test_load_latest_model(self):
        """Test loading the latest model file."""
        processor = ApplicationProcessor()
        
        assert processor.classifier is not None
        assert processor.feature_cols is not None
        assert processor.imputer is not None
        assert processor.scaler is not None
    
    @pytest.mark.skipif(not Path("models").exists(),
                       reason="Model files not available")
    def test_full_processing_pipeline(self, sample_application):
        """Test full processing pipeline with real model."""
        processor = ApplicationProcessor()
        
        result = processor.process_single(sample_application)
        
        assert result['success'] is True
        assert result['predicted_quartile'] in ['Q1', 'Q2', 'Q3', 'Q4']
        assert 0 <= result['confidence'] <= 100
        assert isinstance(result['needs_review'], bool)