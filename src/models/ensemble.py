"""
Ensemble Framework: Combine multiple models for better predictions
Supports weighted averaging, stacking, and confidence-based blending
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle


class ModelEnsemble:
    """
    Ensemble multiple models with different strategies
    """
    
    def __init__(self, models: Dict[str, any] = None):
        """
        Args:
            models: Dictionary mapping model names to model objects
        """
        self.models = models or {}
        self.weights = None
        self.performance_history = {}
        
    def add_model(self, name: str, model: any):
        """Add a model to the ensemble"""
        self.models[name] = model
    
    def predict_all(self, X: any, feature_cols: list = None) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models
        
        Args:
            X: Input data (DataFrame for tabular models, ndarray for neural nets)
            feature_cols: Feature columns (for tabular models)
            
        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Check if model has predict method
                if hasattr(model, 'predict'):
                    if feature_cols is not None and isinstance(X, pd.DataFrame):
                        pred = model.predict(X, feature_cols)
                    else:
                        pred = model.predict(X)
                    predictions[name] = pred
            except Exception as e:
                print(f"Warning: Model {name} failed to predict: {e}")
                
        return predictions
    
    def simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of all predictions"""
        pred_array = np.array(list(predictions.values()))
        return np.mean(pred_array, axis=0)
    
    def weighted_average(self, predictions: Dict[str, np.ndarray], 
                        weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Weighted average of predictions
        
        Args:
            predictions: Dictionary of predictions
            weights: Dictionary of weights (if None, uses equal weights)
        """
        if weights is None:
            return self.simple_average(predictions)
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted sum
        result = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            if name in normalized_weights:
                result += pred * normalized_weights[name]
        
        return result
    
    def confidence_based_ensemble(self, predictions: Dict[str, np.ndarray],
                                  return_confidence: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Ensemble based on model agreement (confidence)
        Models that agree get higher weight
        
        Args:
            predictions: Dictionary of predictions
            return_confidence: Whether to return confidence scores
            
        Returns:
            Ensemble predictions and optionally confidence scores
        """
        pred_array = np.array(list(predictions.values()))
        
        # Calculate mean and std across models
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Confidence is inverse of disagreement (std)
        # Normalize to [0, 1] range
        max_std = np.max(std_pred) if np.max(std_pred) > 0 else 1.0
        confidence = 1.0 - (std_pred / max_std)
        
        if return_confidence:
            return mean_pred, confidence
        else:
            return mean_pred, None
    
    def performance_weighted_ensemble(self, predictions: Dict[str, np.ndarray],
                                     recent_performance: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Weight models by their recent performance
        
        Args:
            predictions: Dictionary of predictions
            recent_performance: Dictionary of recent performance metrics (e.g., correlation)
        """
        if recent_performance is None or len(recent_performance) == 0:
            return self.simple_average(predictions)
        
        # Convert performance to weights (higher is better)
        # Ensure positive weights
        min_perf = min(recent_performance.values())
        shifted_perf = {k: v - min_perf + 0.01 for k, v in recent_performance.items()}
        
        return self.weighted_average(predictions, shifted_perf)
    
    def adaptive_ensemble(self, predictions: Dict[str, np.ndarray],
                         recent_errors: Optional[Dict[str, List[float]]] = None) -> np.ndarray:
        """
        Adaptive ensemble that adjusts weights based on recent errors
        
        Args:
            predictions: Dictionary of predictions
            recent_errors: Dictionary of recent prediction errors for each model
        """
        if recent_errors is None or len(recent_errors) == 0:
            return self.simple_average(predictions)
        
        # Calculate inverse RMSE as weights
        weights = {}
        for name in predictions.keys():
            if name in recent_errors and len(recent_errors[name]) > 0:
                rmse = np.sqrt(np.mean(np.array(recent_errors[name]) ** 2))
                # Inverse RMSE (lower error = higher weight)
                weights[name] = 1.0 / (rmse + 1e-6)
            else:
                weights[name] = 1.0
        
        return self.weighted_average(predictions, weights)
    
    def predict_ensemble(self, X: any, feature_cols: list = None,
                        strategy: str = 'simple',
                        weights: Optional[Dict[str, float]] = None,
                        return_all: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Make ensemble predictions
        
        Args:
            X: Input data
            feature_cols: Feature columns (for tabular models)
            strategy: Ensemble strategy ('simple', 'weighted', 'confidence', 'performance', 'adaptive')
            weights: Custom weights for weighted strategy
            return_all: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions and optionally all individual predictions
        """
        # Get predictions from all models
        all_predictions = self.predict_all(X, feature_cols)
        
        if len(all_predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # Apply ensemble strategy
        if strategy == 'simple':
            ensemble_pred = self.simple_average(all_predictions)
        elif strategy == 'weighted':
            ensemble_pred = self.weighted_average(all_predictions, weights)
        elif strategy == 'confidence':
            ensemble_pred, _ = self.confidence_based_ensemble(all_predictions)
        elif strategy == 'performance':
            ensemble_pred = self.performance_weighted_ensemble(
                all_predictions,
                self.performance_history
            )
        elif strategy == 'adaptive':
            ensemble_pred = self.adaptive_ensemble(all_predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if return_all:
            return ensemble_pred, all_predictions
        else:
            return ensemble_pred, None
    
    def update_performance(self, model_name: str, performance_score: float):
        """
        Update performance history for a model
        
        Args:
            model_name: Name of the model
            performance_score: Performance metric (e.g., correlation, negative RMSE)
        """
        self.performance_history[model_name] = performance_score
    
    def calculate_ensemble_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate confidence score for ensemble predictions based on model agreement
        
        Returns:
            Confidence scores (0-1) for each prediction
        """
        pred_array = np.array(list(predictions.values()))
        
        # Coefficient of variation (CV) = std / mean
        # Lower CV = higher confidence
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Avoid division by zero
        cv = np.abs(std_pred / (mean_pred + 1e-6))
        
        # Normalize to confidence (inverse CV)
        # High CV -> Low confidence, Low CV -> High confidence
        max_cv = np.percentile(cv, 95)  # Use 95th percentile as max
        confidence = 1.0 - np.clip(cv / max_cv, 0, 1)
        
        return confidence
    
    def save(self, filepath: str):
        """Save ensemble configuration"""
        metadata = {
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load ensemble configuration"""
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.weights = metadata.get('weights')
        self.performance_history = metadata.get('performance_history', {})


class StackedEnsemble:
    """
    Stacked ensemble using meta-learner
    Train a secondary model on predictions of base models
    """
    
    def __init__(self, base_models: Dict[str, any], meta_model: any = None):
        """
        Args:
            base_models: Dictionary of base models
            meta_model: Meta-learner model (e.g., linear regression, light GBM)
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_features_names = None
    
    def create_meta_features(self, X: any, feature_cols: list = None) -> pd.DataFrame:
        """
        Create meta-features from base model predictions
        """
        meta_features = {}
        
        for name, model in self.base_models.items():
            try:
                if feature_cols is not None and isinstance(X, pd.DataFrame):
                    pred = model.predict(X, feature_cols)
                else:
                    pred = model.predict(X)
                meta_features[f'{name}_pred'] = pred
            except Exception as e:
                print(f"Warning: Base model {name} failed: {e}")
        
        return pd.DataFrame(meta_features)
    
    def train_meta_model(self, X_train: any, y_train: np.ndarray,
                        feature_cols: list = None):
        """
        Train the meta-learner on base model predictions
        """
        # Get base model predictions
        meta_features = self.create_meta_features(X_train, feature_cols)
        self.meta_features_names = list(meta_features.columns)
        
        print(f"Training meta-model with {len(self.meta_features_names)} meta-features...")
        
        # Train meta-model
        if hasattr(self.meta_model, 'train_full'):
            # For custom models with train_full method
            meta_df = meta_features.copy()
            meta_df['target'] = y_train
            self.meta_model.train_full(meta_df, self.meta_features_names, 'target')
        elif hasattr(self.meta_model, 'fit'):
            # For sklearn-style models
            self.meta_model.fit(meta_features, y_train)
        else:
            raise ValueError("Meta model must have train_full or fit method")
    
    def predict(self, X: any, feature_cols: list = None) -> np.ndarray:
        """Make stacked predictions"""
        # Get base model predictions
        meta_features = self.create_meta_features(X, feature_cols)
        
        # Predict with meta-model
        if hasattr(self.meta_model, 'predict'):
            if isinstance(meta_features, pd.DataFrame):
                return self.meta_model.predict(meta_features, self.meta_features_names)
            else:
                return self.meta_model.predict(meta_features)
        else:
            raise ValueError("Meta model must have predict method")

