"""
Enhanced Kaggle Submission with Ensemble Models
Uses multiple models with confidence-based allocation
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pickle

import kaggle_evaluation.default_inference_server

# Configure paths
MODEL_DIR = Path('/kaggle/input/hull-tactical-models') if Path('/kaggle/input').exists() else Path('models')
DATA_DIR = Path('/kaggle/input/hull-tactical-market-prediction') if Path('/kaggle/input').exists() else Path('.')
sys.path.insert(0, str(MODEL_DIR))


class HullTacticalEnsemblePredictor:
    """
    Enhanced predictor using ensemble of models with confidence-based allocation
    """
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.feature_selector = None
        self.allocator = None
        self.feature_names = None
        self.historical_data = None
        self.test_rows = []
        self.use_ensemble = False
        
        self.load_models()
        self.load_training_data()
        
    def load_models(self):
        """Load all available models"""
        print("Loading models...")
        
        # Load feature engineer
        try:
            with open(MODEL_DIR / 'feature_engineer.pkl', 'rb') as f:
                self.feature_engineer = pickle.load(f)
            print("  ✓ Feature engineer loaded")
        except Exception as e:
            print(f"  ✗ Feature engineer failed: {e}")
            raise
        
        # Load feature selector
        try:
            with open(MODEL_DIR / 'feature_selector.pkl', 'rb') as f:
                self.feature_selector = pickle.load(f)
                self.feature_names = self.feature_selector.selected_features
            print(f"  ✓ Feature selector loaded ({len(self.feature_names)} features)")
        except Exception as e:
            print(f"  ⚠ Feature selector not available: {e}")
            # Fallback to metadata
            try:
                with open(MODEL_DIR / 'model_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                    self.feature_names = metadata['feature_names']
                print(f"  ✓ Loaded feature names from metadata ({len(self.feature_names)} features)")
            except:
                with open(MODEL_DIR / 'gbm_model.txt.metadata', 'rb') as f:
                    metadata = pickle.load(f)
                    self.feature_names = metadata['feature_names']
                print(f"  ✓ Loaded feature names from model metadata ({len(self.feature_names)} features)")
        
        # Load LightGBM (primary model)
        try:
            import lightgbm as lgb
            self.models['lightgbm'] = lgb.Booster(model_file=str(MODEL_DIR / 'gbm_model.txt'))
            print("  ✓ LightGBM loaded")
        except Exception as e:
            print(f"  ✗ LightGBM failed: {e}")
            raise
        
        # Load XGBoost (optional)
        try:
            import xgboost as xgb
            self.models['xgboost'] = xgb.Booster()
            self.models['xgboost'].load_model(str(MODEL_DIR / 'xgb_model.json'))
            print("  ✓ XGBoost loaded")
        except Exception as e:
            print(f"  ⚠ XGBoost not available: {e}")
        
        # Load CatBoost (optional)
        try:
            import catboost as cb
            self.models['catboost'] = cb.CatBoostRegressor()
            self.models['catboost'].load_model(str(MODEL_DIR / 'catboost_model.cbm'))
            print("  ✓ CatBoost loaded")
        except Exception as e:
            print(f"  ⚠ CatBoost not available: {e}")
        
        # Load allocation strategy
        try:
            with open(MODEL_DIR / 'allocation_strategy.pkl', 'rb') as f:
                self.allocator = pickle.load(f)
            print("  ✓ Allocation strategy loaded")
        except Exception as e:
            print(f"  ⚠ Allocation strategy not available: {e}")
            # Create default
            from src.models import AllocationStrategy
            self.allocator = AllocationStrategy()
        
        # Check if ensemble available
        self.use_ensemble = len(self.models) > 1
        if self.use_ensemble:
            print(f"\n✓ Ensemble mode: {len(self.models)} models")
        else:
            print(f"\n✓ Single model mode: LightGBM only")
    
    def load_training_data(self):
        """Load historical training data for feature engineering"""
        print("\nLoading training data...")
        train_path = DATA_DIR / 'train.csv'
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            # Use recent data for rolling window
            self.historical_data = train_df.tail(500).copy()
            print(f"  ✓ Loaded {len(self.historical_data)} historical samples")
        else:
            print("  ⚠ Training data not found")
            self.historical_data = pd.DataFrame()
    
    def get_model_predictions(self, features_df):
        """Get predictions from all available models"""
        predictions = {}
        
        # Prepare features
        X = features_df[self.feature_names].fillna(method='ffill').fillna(0)
        
        # LightGBM
        try:
            pred = self.models['lightgbm'].predict(X)[0]
            predictions['lightgbm'] = pred
        except Exception as e:
            print(f"LightGBM prediction failed: {e}")
        
        # XGBoost
        if 'xgboost' in self.models:
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                pred = self.models['xgboost'].predict(dmatrix)[0]
                predictions['xgboost'] = pred
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")
        
        # CatBoost
        if 'catboost' in self.models:
            try:
                pred = self.models['catboost'].predict(X)[0]
                predictions['catboost'] = pred
            except Exception as e:
                print(f"CatBoost prediction failed: {e}")
        
        return predictions
    
    def calculate_confidence(self, predictions):
        """Calculate confidence based on model agreement"""
        if len(predictions) < 2:
            return 0.85  # Default high confidence for single model
        
        pred_values = list(predictions.values())
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        
        # Lower std = higher confidence
        # Normalize to 0-1 range
        confidence = 1.0 - min(std_pred / (abs(mean_pred) + 0.01), 1.0)
        confidence = np.clip(confidence, 0.3, 0.95)  # Keep in reasonable range
        
        return confidence
    
    def predict_allocation(self, test_row_pl: pl.DataFrame) -> float:
        """
        Main prediction function called by Kaggle evaluator
        
        Args:
            test_row_pl: Polars DataFrame with single test row
            
        Returns:
            Allocation weight in [0, 2]
        """
        # Convert to pandas
        test_row = test_row_pl.to_pandas()
        
        # Add target column if missing (needed for feature engineering)
        if 'market_forward_excess_returns' not in test_row.columns:
            test_row['market_forward_excess_returns'] = 0.0
        
        # Store test row for rolling window
        self.test_rows.append(test_row.copy())
        
        # Combine with historical data
        if self.historical_data is not None and len(self.historical_data) > 0:
            combined = pd.concat([self.historical_data] + self.test_rows, ignore_index=True)
        else:
            combined = pd.concat(self.test_rows, ignore_index=True) if len(self.test_rows) > 1 else test_row
        
        # Keep only recent data (memory management)
        if len(combined) > 500:
            combined = combined.tail(500)
        
        try:
            # Engineer features
            features = self.feature_engineer.create_all_tabular_features(
                combined,
                target_col='market_forward_excess_returns'
            )
            
            # Get last row (current test sample)
            features_row = features.iloc[-1:]
            
            # Ensure all required features exist
            for feat in self.feature_names:
                if feat not in features_row.columns:
                    features_row[feat] = 0
            
            # Get predictions from all models
            predictions = self.get_model_predictions(features_row)
            
            if len(predictions) == 0:
                print("Warning: No predictions available, using neutral allocation")
                return 1.0
            
            # Calculate ensemble prediction
            ensemble_prediction = np.mean(list(predictions.values()))
            
            # Calculate confidence
            confidence = self.calculate_confidence(predictions)
            
            # Get market volatility from historical data
            if len(combined) > 20:
                recent_returns = combined['market_forward_excess_returns'].dropna().tail(252)
                if len(recent_returns) > 10:
                    market_volatility = recent_returns.std() * np.sqrt(252)
                else:
                    market_volatility = 0.15  # Default
            else:
                market_volatility = 0.15
            
            # Calculate allocation using simple method (more sensitive)
            # Use simple_allocation which actually varies based on predictions
            allocation = self.allocator.simple_allocation(
                predicted_return=ensemble_prediction,
                predicted_sign_confidence=confidence
            )
            
            # Ensure valid range
            allocation = float(np.clip(allocation, 0.0, 2.0))
            
            # Log every 100 predictions
            if len(self.test_rows) % 100 == 0:
                print(f"\nPrediction #{len(self.test_rows)}:")
                print(f"  Models: {list(predictions.keys())}")
                print(f"  Predictions: {predictions}")
                print(f"  Ensemble: {ensemble_prediction:.6f}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Allocation: {allocation:.4f}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            allocation = 1.0  # Neutral fallback
        
        return allocation


# Global predictor instance
predictor = None

def predict(test: pl.DataFrame) -> float:
    """
    Prediction function called by Kaggle evaluator
    
    Args:
        test: Polars DataFrame with single test row
        
    Returns:
        Allocation weight in [0, 2]
    """
    global predictor
    if predictor is None:
        print("=" * 80)
        print("INITIALIZING HULL TACTICAL ENSEMBLE PREDICTOR")
        print("=" * 80)
        predictor = HullTacticalEnsemblePredictor()
        print("\n✓ Predictor ready!")
        print("=" * 80)
    
    return predictor.predict_allocation(test)


# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))

