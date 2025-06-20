"""
Ordinal Regression Model for Medical Admissions
==============================================

Implements XGBoost with custom ordinal objective function for 
4-bucket classification with natural break boundaries.

Buckets:
1. Reject (0-9): 13.6%
2. Waitlist (11-15): 33.7%  
3. Interview (17-21): 35.6%
4. Accept (23-25): 17.2%
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import joblib
from scipy.special import expit  # Sigmoid function
import warnings
warnings.filterwarnings('ignore')


class OrdinalXGBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with custom ordinal regression objective.
    Uses cumulative link model for proper ordinal classification.
    """
    
    def __init__(self, 
                 n_classes: int = 4,
                 bucket_boundaries: List[float] = [0, 10, 16, 22, 26],
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.02,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        
        self.n_classes = n_classes
        self.bucket_boundaries = bucket_boundaries
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        # Will be set during fit
        self.models_ = []  # One model per threshold
        self.thresholds_ = None
        self.feature_names_ = None
        
    def _create_ordinal_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Convert bucket labels to ordinal targets.
        For K classes, create K-1 binary targets.
        """
        n_samples = len(y)
        n_thresholds = self.n_classes - 1
        
        # Create binary targets for each threshold
        ordinal_y = np.zeros((n_samples, n_thresholds))
        
        for i in range(n_thresholds):
            ordinal_y[:, i] = (y > i).astype(int)
            
        return ordinal_y
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[List[Tuple]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose: bool = True):
        """
        Fit ordinal regression model.
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        
        # Convert to ordinal targets
        ordinal_y = self._create_ordinal_targets(y)
        self.thresholds_ = list(range(1, self.n_classes))
        
        # Train one model per threshold
        self.models_ = []
        
        for i, threshold in enumerate(self.thresholds_):
            if verbose:
                print(f"\nTraining model for threshold {threshold} (>{i})...")
            
            # Create model for this threshold
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state + i,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Prepare eval set if provided
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                ordinal_y_val = self._create_ordinal_targets(y_val)
                eval_set_i = [(X_val, ordinal_y_val[:, i])]
            else:
                eval_set_i = None
            
            # Fit model
            model.fit(
                X, ordinal_y[:, i],
                eval_set=eval_set_i,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            self.models_.append(model)
            
            if verbose:
                # Calculate training accuracy for this threshold
                train_pred = model.predict(X)
                acc = np.mean(train_pred == ordinal_y[:, i])
                print(f"  Training accuracy: {acc:.3f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of each class using cumulative link model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        n_thresholds = len(self.models_)
        
        # Get cumulative probabilities P(Y > k)
        cumulative_probs = np.zeros((n_samples, n_thresholds + 1))
        cumulative_probs[:, 0] = 1.0  # P(Y > -1) = 1
        
        for i, model in enumerate(self.models_):
            # Get P(Y > i)
            cumulative_probs[:, i + 1] = model.predict_proba(X)[:, 1]
        
        # Convert to class probabilities P(Y = k) = P(Y > k-1) - P(Y > k)
        class_probs = np.zeros((n_samples, self.n_classes))
        
        for k in range(self.n_classes):
            if k == 0:
                class_probs[:, k] = 1 - cumulative_probs[:, 1]
            elif k == self.n_classes - 1:
                class_probs[:, k] = cumulative_probs[:, k]
            else:
                class_probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k + 1]
        
        # Ensure probabilities sum to 1 and are non-negative
        class_probs = np.maximum(class_probs, 0)
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
        
        return class_probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels with confidence scores.
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        
        return predictions, confidences
    
    def score_to_bucket(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert continuous scores to bucket labels.
        """
        buckets = np.digitize(scores, self.bucket_boundaries[1:-1])
        return buckets
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get averaged feature importance across all threshold models.
        """
        if not self.feature_names_:
            return {}
        
        # Average importance across all models
        importance_sum = np.zeros(len(self.feature_names_))
        
        for model in self.models_:
            importance_sum += model.feature_importances_
            
        avg_importance = importance_sum / len(self.models_)
        
        # Create dictionary
        importance_dict = {
            feature: importance 
            for feature, importance in zip(self.feature_names_, avg_importance)
        }
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def calculate_quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Quadratic Weighted Kappa for ordinal classification.
    """
    n_classes = max(max(y_true), max(y_pred)) + 1
    weights = np.zeros((n_classes, n_classes))
    
    # Create weight matrix
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2
    
    # Normalize weights
    weights = weights / (n_classes - 1) ** 2
    
    # Calculate observed and expected matrices
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    n_samples = np.sum(conf_mat)
    
    # Observed agreement
    po = np.sum(conf_mat * (1 - weights)) / n_samples
    
    # Expected agreement
    expected = np.outer(conf_mat.sum(axis=1), conf_mat.sum(axis=0)) / n_samples
    pe = np.sum(expected * (1 - weights)) / n_samples
    
    # Kappa
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0
    
    return kappa


def evaluate_ordinal_model(model: OrdinalXGBoostClassifier, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray,
                          bucket_names: List[str] = None) -> Dict:
    """
    Comprehensive evaluation of ordinal model.
    """
    if bucket_names is None:
        bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
    
    # Get predictions
    y_pred, confidences = model.predict_with_confidence(X_test)
    
    # Calculate metrics
    exact_accuracy = np.mean(y_pred == y_test)
    adjacent_accuracy = np.mean(np.abs(y_pred - y_test) <= 1)
    qwk = calculate_quadratic_weighted_kappa(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-bucket metrics
    bucket_metrics = {}
    for i, name in enumerate(bucket_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            bucket_acc = np.mean(y_pred[mask] == i)
            bucket_conf = np.mean(confidences[mask])
            bucket_metrics[name] = {
                'accuracy': bucket_acc,
                'avg_confidence': bucket_conf,
                'count': np.sum(mask)
            }
    
    results = {
        'exact_accuracy': exact_accuracy,
        'adjacent_accuracy': adjacent_accuracy,
        'quadratic_weighted_kappa': qwk,
        'confusion_matrix': cm,
        'bucket_metrics': bucket_metrics,
        'avg_confidence': np.mean(confidences)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("ORDINAL MODEL EVALUATION")
    print("="*60)
    print(f"\n✓ Overall Metrics:")
    print(f"  Exact Match Accuracy: {exact_accuracy:.1%}")
    print(f"  Adjacent Accuracy (±1): {adjacent_accuracy:.1%}")
    print(f"  Quadratic Weighted Kappa: {qwk:.3f}")
    print(f"  Average Confidence: {np.mean(confidences):.1%}")
    
    print(f"\n✓ Confusion Matrix:")
    print(f"{'':>12s}", end='')
    for name in bucket_names:
        print(f"{name:>12s}", end='')
    print()
    
    for i, name in enumerate(bucket_names):
        print(f"{name:>12s}", end='')
        for j in range(len(bucket_names)):
            print(f"{cm[i, j]:>12d}", end='')
        print()
    
    print(f"\n✓ Per-Bucket Performance:")
    for name, metrics in bucket_metrics.items():
        print(f"  {name}: {metrics['accuracy']:.1%} accuracy, "
              f"{metrics['avg_confidence']:.1%} avg confidence "
              f"(n={metrics['count']})")
    
    return results


# Example usage
if __name__ == "__main__":
    print("Ordinal Regression Model for Medical Admissions")
    print("="*60)
    
    # Example with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic ordinal data
    X = np.random.randn(n_samples, n_features)
    # Create ordinal target with some structure
    y_continuous = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples)
    y = pd.qcut(y_continuous, q=4, labels=[0, 1, 2, 3]).values
    
    # Split data
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.setdiff1d(range(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = OrdinalXGBoostClassifier()
    model.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    results = evaluate_ordinal_model(model, X_test, y_test)
    
    print("\n✓ Model saved as: ordinal_model_demo.pkl")
    joblib.dump(model, 'ordinal_model_demo.pkl')