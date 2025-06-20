"""Model loading utilities."""

import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading and management."""
    
    def __init__(self, models_dir: Path = None):
        """Initialize model loader.
        
        Args:
            models_dir: Directory containing model files
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = models_dir
    
    def load_model(self, model_path: Path) -> Dict[str, Any]:
        """Load a specific model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary containing model components
        """
        try:
            model_data = joblib.load(model_path)
            
            # Create cascade classifier from optimizer
            if 'optimizer' in model_data:
                optimizer = model_data['optimizer']
                models = {
                    'stage1': optimizer.best_models.get('stage1'),
                    'stage2': optimizer.best_models.get('stage2'),
                    'stage3': optimizer.best_models.get('stage3')
                }
                
                from .cascade_classifier import CascadeClassifier
                classifier = CascadeClassifier(models)
            else:
                classifier = model_data.get('classifier')
            
            return {
                'classifier': classifier,
                'feature_cols': model_data.get('feature_cols', []),
                'imputer': model_data.get('imputer'),
                'scaler': model_data.get('scaler'),
                'performance': model_data.get('performance', {}),
                'metadata': {
                    'path': str(model_path),
                    'name': model_path.name
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_latest_model(self) -> Optional[Dict[str, Any]]:
        """Load the latest model from the models directory.
        
        Returns:
            Model data dictionary or None if no models found
        """
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return None
        
        # Find all model files
        model_files = list(self.models_dir.glob("*.pkl"))
        
        if not model_files:
            logger.warning("No model files found")
            return None
        
        # Sort by modification time and get latest
        latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
        
        logger.info(f"Loading latest model: {latest_model.name}")
        return self.load_model(latest_model)
    
    def list_available_models(self) -> list:
        """List all available models.
        
        Returns:
            List of model file paths
        """
        if not self.models_dir.exists():
            return []
        
        model_files = list(self.models_dir.glob("*.pkl"))
        return sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)