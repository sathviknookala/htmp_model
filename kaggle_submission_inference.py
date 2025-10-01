"""
Kaggle Submission Script for Hull Tactical Market Prediction
This script uses the InferenceServer pattern required for forecasting competitions
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

# Model paths
MODEL_DIR = Path('/kaggle/input/hull-tactical-models') if Path('/kaggle/input').exists() else Path('models')

# Add MODEL_DIR to Python path so src modules can be found
sys.path.insert(0, str(MODEL_DIR))
if (MODEL_DIR / 'src').exists():
    print(f"Added {MODEL_DIR} to Python path (src module found)")


class HullTacticalPredictor:
    """Main predictor class"""
    
    def __init__(self):
        self.feature_engineer = None
        self.gbm_model = None
        self.allocation_strategy = None
        self.feature_names = None
        self.load_models()
        
    def load_models(self):
        """Load trained models and feature engineer"""
        try:
            # Load feature engineer
            feature_engineer_path = MODEL_DIR / 'feature_engineer.pkl'
            with open(feature_engineer_path, 'rb') as f:
                self.feature_engineer = pickle.load(f)
            print("✓ Loaded feature engineer")
            
            # Load GBM model
            gbm_model_path = MODEL_DIR / 'gbm_model.txt'
            self.gbm_model = lgb.Booster(model_file=str(gbm_model_path))
            print("✓ Loaded GBM model")
            
            # Load metadata
            metadata_path = MODEL_DIR / 'gbm_model.txt.metadata'
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.feature_names = metadata['feature_names']
            print(f"✓ Loaded {len(self.feature_names)} features")
            
            # Load allocation strategy
            allocation_strategy_path = MODEL_DIR / 'allocation_strategy.pkl'
            with open(allocation_strategy_path, 'rb') as f:
                self.allocation_strategy = pickle.load(f)
            print("✓ Loaded allocation strategy")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def create_features(self, data: pd.DataFrame, historical_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create features for prediction"""
        if historical_data is not None and len(historical_data) > 0:
            combined = pd.concat([historical_data, data], ignore_index=False)
        else:
            combined = data
        
        features = self.feature_engineer.create_all_tabular_features(
            combined,
            target_col='market_forward_excess_returns'
        )
        
        return features.iloc[-len(data):]
    
    def predict_batch(self, test_batch: pd.DataFrame, historical_data: pd.DataFrame = None) -> pd.DataFrame:
        """Make predictions for a batch"""
        # Create features
        features_df = self.create_features(test_batch, historical_data)
        
        # Handle missing features
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        for feat in missing_features:
            features_df[feat] = 0
        
        # Get features in correct order
        X = features_df[self.feature_names].copy()
        X = X.ffill().fillna(0)
        
        # Make predictions
        predictions = self.gbm_model.predict(X)
        
        # Convert to allocations
        allocations = self.convert_to_allocations(predictions, test_batch)
        
        # Create result
        result = pd.DataFrame({
            'batch_id': test_batch['batch_id'] if 'batch_id' in test_batch.columns else test_batch['date_id'],
            'prediction': allocations
        })
        
        return result
    
    def convert_to_allocations(self, predictions: np.ndarray, test_batch: pd.DataFrame) -> np.ndarray:
        """Convert return predictions to allocation weights [0, 2]"""
        allocations = []
        
        for pred in predictions:
            if pred > 0:
                allocation = 1.0 + min(pred / 0.01, 1.0)
            elif pred < 0:
                allocation = 1.0 + max(pred / 0.01, -1.0)
            else:
                allocation = 1.0
            
            allocation = np.clip(allocation, 0.0, 2.0)
            allocations.append(allocation)
        
        return np.array(allocations)


# Initialize predictor globally
predictor = None


def predict(test: pd.DataFrame, sample_submission: pd.DataFrame = None, 
           sample_prediction: pd.DataFrame = None, prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main prediction function called by Kaggle evaluation framework
    
    Args:
        test: Test data batch (polars or pandas DataFrame)
        sample_submission: Sample submission format (optional)
        sample_prediction: Sample prediction format (optional)  
        prices: Price data (optional)
        
    Returns:
        DataFrame with predictions
    """
    global predictor
    
    # Initialize predictor on first call
    if predictor is None:
        print("Initializing predictor...")
        predictor = HullTacticalPredictor()
        print("Predictor ready!")
    
    # Convert polars to pandas if needed
    if hasattr(test, 'to_pandas'):
        test = test.to_pandas()
    
    # Make predictions
    result = predictor.predict_batch(test, historical_data=None)
    
    return result


# For forecasting competitions using InferenceServer
if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in Kaggle competition mode...")
        from kaggle_evaluation.core.templates import InferenceServer
        
        class MyInferenceServer(InferenceServer):
            def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
                # Not used in actual competition
                pass
        
        # Start the inference server
        inference_server = MyInferenceServer(predict)
        inference_server.serve()
    else:
        # Test locally
        print("Testing locally...")
        
        if os.path.exists('test.csv') and os.path.exists('train.csv'):
            test_df = pd.read_csv('test.csv')
            train_df = pd.read_csv('train.csv')
            
            print(f"Test shape: {test_df.shape}")
            print(f"Train shape: {train_df.shape}")
            
            # Test on first batch
            if 'batch_id' in test_df.columns:
                first_batch = test_df[test_df['batch_id'] == test_df['batch_id'].iloc[0]]
            else:
                first_batch = test_df.head(1)
            
            result = predict(first_batch)
            print("\nPrediction:")
            print(result)
            print(f"\nRange: [{result['prediction'].min():.4f}, {result['prediction'].max():.4f}]")
        else:
            print("Local test files not found. Script ready for Kaggle.")

