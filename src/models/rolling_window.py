"""
Rolling Window Retraining Framework
Retrain models on expanding window during forecasting phase
"""
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List, Any
import pickle
from pathlib import Path


class RollingWindowTrainer:
    """
    Framework for rolling window model retraining
    """
    
    def __init__(self, model_factory: Callable,
                 feature_engineer: Any,
                 window_type: str = 'expanding',
                 min_train_size: int = 1000,
                 retrain_frequency: int = 50):
        """
        Args:
            model_factory: Function that creates a new model instance
            feature_engineer: Feature engineering object
            window_type: 'expanding' or 'rolling'
            min_train_size: Minimum training set size
            retrain_frequency: Retrain every N samples
        """
        self.model_factory = model_factory
        self.feature_engineer = feature_engineer
        self.window_type = window_type
        self.min_train_size = min_train_size
        self.retrain_frequency = retrain_frequency
        
        self.historical_data = []
        self.models = []
        self.predictions = []
        self.actuals = []
        self.retrain_points = []
        
    def should_retrain(self, current_size: int) -> bool:
        """
        Determine if model should be retrained
        
        Args:
            current_size: Current size of historical data
            
        Returns:
            True if should retrain
        """
        if current_size < self.min_train_size:
            return False
        
        if len(self.models) == 0:
            return True
        
        samples_since_retrain = current_size - self.retrain_points[-1]
        return samples_since_retrain >= self.retrain_frequency
    
    def retrain_model(self, train_data: pd.DataFrame,
                     feature_cols: List[str],
                     target_col: str = 'market_forward_excess_returns',
                     **train_kwargs) -> Any:
        """
        Retrain model on current data
        
        Args:
            train_data: Training DataFrame
            feature_cols: Feature columns
            target_col: Target column
            **train_kwargs: Additional training arguments
            
        Returns:
            Trained model
        """
        print(f"\nRetraining model on {len(train_data)} samples...")
        
        # Create new model
        model = self.model_factory()
        
        # Train
        if hasattr(model, 'train_full'):
            model.train_full(train_data, feature_cols, target_col, **train_kwargs)
        elif hasattr(model, 'fit'):
            X = train_data[feature_cols].fillna(method='ffill').fillna(0)
            y = train_data[target_col].dropna()
            model.fit(X, y)
        else:
            raise ValueError("Model must have train_full or fit method")
        
        return model
    
    def predict_with_rolling_window(self, test_data: pd.DataFrame,
                                    feature_cols: List[str],
                                    target_col: str = 'market_forward_excess_returns',
                                    initial_train_data: Optional[pd.DataFrame] = None,
                                    **train_kwargs) -> pd.DataFrame:
        """
        Make predictions with rolling window retraining
        
        Args:
            test_data: Test DataFrame
            feature_cols: Feature columns
            target_col: Target column
            initial_train_data: Initial training data
            **train_kwargs: Additional training arguments
            
        Returns:
            DataFrame with predictions and actuals
        """
        print("=" * 60)
        print("ROLLING WINDOW PREDICTION")
        print("=" * 60)
        print(f"Test samples: {len(test_data)}")
        print(f"Window type: {self.window_type}")
        print(f"Retrain frequency: {self.retrain_frequency}")
        print("=" * 60)
        
        # Initialize with training data
        if initial_train_data is not None:
            self.historical_data = [initial_train_data.copy()]
            current_data = initial_train_data.copy()
        else:
            current_data = pd.DataFrame()
        
        predictions = []
        actuals = []
        
        for i in range(len(test_data)):
            test_row = test_data.iloc[i:i+1]
            
            # Check if we should retrain
            if self.should_retrain(len(current_data)):
                model = self.retrain_model(
                    current_data,
                    feature_cols,
                    target_col,
                    **train_kwargs
                )
                self.models.append(model)
                self.retrain_points.append(len(current_data))
                print(f"  Retrained at sample {i}/{len(test_data)}")
            
            # Use most recent model
            if len(self.models) > 0:
                model = self.models[-1]
                
                # Make prediction
                try:
                    if hasattr(model, 'predict'):
                        # Prepare features
                        combined = pd.concat([current_data, test_row], ignore_index=True)
                        features = self.feature_engineer.create_all_tabular_features(
                            combined,
                            target_col=target_col
                        )
                        features_row = features.iloc[-1:]
                        
                        # Ensure all required features exist
                        for feat in feature_cols:
                            if feat not in features_row.columns:
                                features_row[feat] = 0
                        
                        X = features_row[feature_cols].fillna(method='ffill').fillna(0)
                        pred = float(model.predict(X, feature_cols)[0] if hasattr(model.predict(X, feature_cols), '__iter__') else model.predict(X, feature_cols))
                    else:
                        pred = 0.0
                except Exception as e:
                    print(f"  Warning: Prediction failed at {i}: {e}")
                    pred = 0.0
            else:
                pred = 0.0
            
            predictions.append(pred)
            
            # Get actual value if available
            if target_col in test_row.columns:
                actual = test_row[target_col].values[0]
                actuals.append(actual)
            else:
                actuals.append(np.nan)
            
            # Update historical data
            if self.window_type == 'expanding':
                # Add to growing dataset
                current_data = pd.concat([current_data, test_row], ignore_index=True)
            else:  # rolling
                # Keep fixed window size
                current_data = pd.concat([current_data, test_row], ignore_index=True)
                if len(current_data) > self.min_train_size:
                    current_data = current_data.tail(self.min_train_size)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'actual': actuals
        })
        
        # Calculate metrics if actuals available
        valid_mask = ~np.isnan(results['actual'])
        if valid_mask.sum() > 0:
            pred_valid = results.loc[valid_mask, 'prediction']
            actual_valid = results.loc[valid_mask, 'actual']
            
            rmse = np.sqrt(np.mean((actual_valid - pred_valid) ** 2))
            corr = np.corrcoef(actual_valid, pred_valid)[0, 1]
            
            print("\n" + "=" * 60)
            print("ROLLING WINDOW RESULTS")
            print("=" * 60)
            print(f"RMSE: {rmse:.6f}")
            print(f"Correlation: {corr:.4f}")
            print(f"Models trained: {len(self.models)}")
            print(f"Retrain points: {self.retrain_points}")
            print("=" * 60)
        
        return results
    
    def save(self, directory: str):
        """Save rolling window trainer state"""
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save models
        for i, model in enumerate(self.models):
            if hasattr(model, 'save'):
                model.save(str(directory / f'model_{i}'))
        
        # Save metadata
        metadata = {
            'window_type': self.window_type,
            'min_train_size': self.min_train_size,
            'retrain_frequency': self.retrain_frequency,
            'retrain_points': self.retrain_points,
            'num_models': len(self.models)
        }
        
        with open(directory / 'rolling_window_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, directory: str):
        """Load rolling window trainer state"""
        directory = Path(directory)
        
        with open(directory / 'rolling_window_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.window_type = metadata['window_type']
        self.min_train_size = metadata['min_train_size']
        self.retrain_frequency = metadata['retrain_frequency']
        self.retrain_points = metadata['retrain_points']


class AdaptiveWindowTrainer:
    """
    Adaptive window trainer that adjusts based on performance
    """
    
    def __init__(self, model_factory: Callable,
                 performance_threshold: float = 0.5):
        """
        Args:
            model_factory: Function that creates a new model instance
            performance_threshold: Retrain if performance drops below this
        """
        self.model_factory = model_factory
        self.performance_threshold = performance_threshold
        self.current_model = None
        self.recent_performance = []
        self.performance_window = 50
    
    def evaluate_recent_performance(self) -> float:
        """
        Calculate recent model performance
        
        Returns:
            Performance metric (e.g., correlation)
        """
        if len(self.recent_performance) < 10:
            return 1.0  # Not enough data
        
        recent = self.recent_performance[-self.performance_window:]
        return np.mean(recent)
    
    def should_retrain_adaptive(self) -> bool:
        """
        Decide if retraining is needed based on performance
        
        Returns:
            True if should retrain
        """
        if self.current_model is None:
            return True
        
        performance = self.evaluate_recent_performance()
        return performance < self.performance_threshold
    
    def update_performance(self, prediction: float, actual: float):
        """
        Update performance tracking
        
        Args:
            prediction: Model prediction
            actual: Actual value
        """
        # Calculate directional accuracy
        correct = (prediction * actual) > 0
        self.recent_performance.append(1.0 if correct else 0.0)
        
        # Keep only recent window
        if len(self.recent_performance) > self.performance_window * 2:
            self.recent_performance = self.recent_performance[-self.performance_window:]

