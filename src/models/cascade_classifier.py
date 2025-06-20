"""Cascade classifier for medical school applications."""

import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CascadeClassifier:
    """Three-stage cascade classifier for application ranking.
    
    The cascade makes three binary decisions:
    1. Reject vs Non-Reject (score ≤9 vs >9)
    2. Waitlist vs Higher (score 10-15 vs >15)
    3. Interview vs Accept (score 16-22 vs ≥23)
    """
    
    def __init__(self, models: Dict[str, Any]):
        """Initialize with trained models.
        
        Args:
            models: Dictionary containing stage1, stage2, and stage3 models
        """
        self.models = models
        self.quartile_map = {0: 'Q4', 1: 'Q3', 2: 'Q2', 3: 'Q1'}
        
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence scores.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, probabilities, confidences)
            - predictions: Quartile assignments (0-3)
            - probabilities: Probability matrix (n_samples, 4)
            - confidences: Confidence scores (0-100)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        probabilities = np.zeros((n_samples, 4))
        confidences = np.zeros(n_samples)
        
        # Stage 1: Reject vs Non-Reject
        stage1_proba = self.models['stage1'].predict_proba(X)
        prob_not_reject = stage1_proba[:, 1]
        
        # Calculate confidence using margin and entropy
        prob_margin = np.abs(prob_not_reject - 0.5) * 2
        entropy = -prob_not_reject * np.log(prob_not_reject + 1e-10) - \
                  (1 - prob_not_reject) * np.log(1 - prob_not_reject + 1e-10)
        entropy_norm = 1 - entropy / np.log(2)
        stage1_confidence = (prob_margin + entropy_norm) / 2
        
        # Reject decisions
        is_reject = prob_not_reject < 0.5
        predictions[is_reject] = 0  # Q4
        probabilities[is_reject, 0] = 1 - prob_not_reject[is_reject]
        confidences[is_reject] = stage1_confidence[is_reject]
        
        # Non-reject: proceed to stage 2
        non_reject_mask = ~is_reject
        if non_reject_mask.sum() > 0:
            X_stage2 = X[non_reject_mask]
            
            # Stage 2: Waitlist vs Higher
            stage2_proba = self.models['stage2'].predict_proba(X_stage2)
            prob_higher = stage2_proba[:, 1]
            
            # Stage 2 confidence
            prob_margin2 = np.abs(prob_higher - 0.5) * 2
            entropy2 = -prob_higher * np.log(prob_higher + 1e-10) - \
                       (1 - prob_higher) * np.log(1 - prob_higher + 1e-10)
            entropy_norm2 = 1 - entropy2 / np.log(2)
            stage2_confidence = (prob_margin2 + entropy_norm2) / 2
            
            # Waitlist decisions
            is_waitlist = prob_higher < 0.5
            waitlist_indices = np.where(non_reject_mask)[0][is_waitlist]
            predictions[waitlist_indices] = 1  # Q3
            probabilities[waitlist_indices, 1] = 1 - prob_higher[is_waitlist]
            
            # Combine confidences (harmonic mean)
            non_reject_conf = confidences[non_reject_mask]
            stage1_conf_subset = stage1_confidence[non_reject_mask]
            non_reject_conf[is_waitlist] = 2 / (1/stage1_conf_subset[is_waitlist] + 
                                                1/stage2_confidence[is_waitlist])
            confidences[non_reject_mask] = non_reject_conf
            
            # Higher: proceed to stage 3
            higher_mask = np.zeros(n_samples, dtype=bool)
            higher_indices = np.where(non_reject_mask)[0][~is_waitlist]
            higher_mask[higher_indices] = True
            
            if higher_mask.sum() > 0:
                X_stage3 = X[higher_mask]
                
                # Stage 3: Interview vs Accept
                stage3_proba = self.models['stage3'].predict_proba(X_stage3)
                prob_accept = stage3_proba[:, 1]
                
                # Stage 3 confidence
                prob_margin3 = np.abs(prob_accept - 0.5) * 2
                entropy3 = -prob_accept * np.log(prob_accept + 1e-10) - \
                          (1 - prob_accept) * np.log(1 - prob_accept + 1e-10)
                entropy_norm3 = 1 - entropy3 / np.log(2)
                stage3_confidence = (prob_margin3 + entropy_norm3) / 2
                
                # Accept/Interview decisions
                is_accept = prob_accept >= 0.5
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                predictions[accept_indices] = 3  # Q1
                predictions[interview_indices] = 2  # Q2
                
                probabilities[accept_indices, 3] = prob_accept[is_accept]
                probabilities[interview_indices, 2] = 1 - prob_accept[~is_accept]
                
                # Final confidence (harmonic mean of all stages)
                higher_conf = confidences[higher_mask]
                stage1_conf_higher = stage1_confidence[higher_mask]
                stage2_conf_higher = stage2_confidence[~is_waitlist]
                
                # Three-way harmonic mean
                higher_conf = 3 / (1/stage1_conf_higher + 1/stage2_conf_higher + 
                                  1/stage3_confidence)
                confidences[higher_mask] = higher_conf
        
        # Fill in remaining probabilities
        for i in range(n_samples):
            prob_sum = probabilities[i].sum()
            if prob_sum < 0.99:
                probabilities[i, predictions[i]] = 1 - prob_sum
        
        # Convert confidence to 0-100 scale with sigmoid transformation
        confidences = 100 / (1 + np.exp(-8 * (confidences - 0.5)))
        
        return predictions, probabilities, confidences
    
    def predict_quartiles(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict quartiles with additional metadata.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions, confidences, and metadata
        """
        predictions, probabilities, confidences = self.predict_with_confidence(X)
        
        # Convert to quartile labels
        quartile_labels = [self.quartile_map[p] for p in predictions]
        
        # Determine review needs
        needs_review = (confidences < 80) | (predictions == 1) | (predictions == 2)
        
        return {
            'quartiles': quartile_labels,
            'numeric_predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences,
            'needs_review': needs_review,
            'statistics': {
                'mean_confidence': float(np.mean(confidences)),
                'low_confidence_pct': float(np.mean(confidences < 60) * 100),
                'high_confidence_pct': float(np.mean(confidences >= 80) * 100)
            }
        }