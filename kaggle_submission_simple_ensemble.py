"""
SIMPLE Ensemble Submission - Direct allocation mapping
Uses ensemble models but with proven simple allocation logic
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

class SimpleEnsemblePredictor:
    """Simple ensemble predictor with direct allocation mapping"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.feature_names = None
        self.historical_data = None
        self.test_rows = []
        self.load_models()
        self.load_training_data()
        
    def load_models(self):
        """Load ensemble models"""
        print("Loading models...")
        
        # Load feature engineer
        with open(MODEL_DIR / 'feature_engineer.pkl', 'rb') as f:
            self.feature_engineer = pickle.load(f)
        print("  ✓ Feature engineer loaded")
        
        # Load feature names from metadata
        with open(MODEL_DIR / 'model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.feature_names = metadata['feature_names']
        print(f"  ✓ Loaded {len(self.feature_names)} features")
        
        # Load LightGBM
        import lightgbm as lgb
        self.models['lightgbm'] = lgb.Booster(model_file=str(MODEL_DIR / 'gbm_model.txt'))
        print("  ✓ LightGBM loaded")
        
        # Load XGBoost
        try:
            import xgboost as xgb
            self.models['xgboost'] = xgb.Booster()
            self.models['xgboost'].load_model(str(MODEL_DIR / 'xgb_model.json'))
            print("  ✓ XGBoost loaded")
        except:
            print("  ⚠ XGBoost not available")
        
        print(f"\n✓ Loaded {len(self.models)} models")
    
    def load_training_data(self):
        """Load historical training data"""
        print("Loading training data...")
        train_path = DATA_DIR / 'train.csv'
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            self.historical_data = train_df.tail(500).copy()
            print(f"  ✓ Loaded {len(self.historical_data)} historical samples\n")
        else:
            print("  ⚠ Training data not found\n")
            self.historical_data = pd.DataFrame()
        
        print("✓ Predictor ready!")
    
    def predict_allocation(self, test_row_pl: pl.DataFrame) -> float:
        """
        Main prediction function
        
        Args:
            test_row_pl: Polars DataFrame with single test row
            
        Returns:
            Allocation weight in [0, 2]
        """
        # Convert to pandas
        test_row = test_row_pl.to_pandas()
        
        # Add target column if missing
        if 'market_forward_excess_returns' not in test_row.columns:
            test_row['market_forward_excess_returns'] = 0.0
        
        # Store test row
        self.test_rows.append(test_row.copy())
        
        # Combine with historical data
        if self.historical_data is not None and len(self.historical_data) > 0:
            combined = pd.concat([self.historical_data] + self.test_rows, ignore_index=True)
        else:
            combined = pd.concat(self.test_rows, ignore_index=True) if len(self.test_rows) > 1 else test_row
        
        # Keep only recent data
        if len(combined) > 500:
            combined = combined.tail(500)
        
        try:
            # Engineer features
            features = self.feature_engineer.create_all_tabular_features(
                combined,
                target_col='market_forward_excess_returns'
            )
            
            # Get last row
            features_row = features.iloc[-1:]
            
            # Ensure all features exist
            for feat in self.feature_names:
                if feat not in features_row.columns:
                    features_row[feat] = 0
            
            # Prepare X
            X = features_row[self.feature_names].ffill().fillna(0)
            
            # Get predictions from all models
            predictions = []
            
            # LightGBM
            try:
                pred = float(self.models['lightgbm'].predict(X)[0])
                predictions.append(pred)
            except Exception as e:
                print(f"LightGBM error: {e}")
            
            # XGBoost
            if 'xgboost' in self.models:
                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(X)
                    pred = float(self.models['xgboost'].predict(dmatrix)[0])
                    predictions.append(pred)
                except Exception as e:
                    print(f"XGBoost error: {e}")
            
            # Ensemble prediction (average)
            if len(predictions) == 0:
                return 1.0  # Neutral if no predictions
            
            ensemble_prediction = np.mean(predictions)
            
            # SIMPLE DIRECT MAPPING
            # Scale factor: 0.01 means 1% predicted return → full swing
            scale_factor = 0.01
            
            if ensemble_prediction > 0:
                allocation = 1.0 + min(ensemble_prediction / scale_factor, 1.0)
            else:
                allocation = 1.0 + max(ensemble_prediction / scale_factor, -1.0)
            
            # Clip to valid range
            allocation = float(np.clip(allocation, 0.0, 2.0))
            
            # Log every 100 predictions
            if len(self.test_rows) % 100 == 0:
                print(f"\nPrediction #{len(self.test_rows)}:")
                print(f"  Predictions: {predictions}")
                print(f"  Ensemble: {ensemble_prediction:.6f}")
                print(f"  Allocation: {allocation:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            allocation = 1.0  # Neutral fallback
        
        return allocation


# Global predictor
predictor = None

def predict(test: pl.DataFrame) -> float:
    """Prediction function called by Kaggle"""
    global predictor
    if predictor is None:
        print("=" * 70)
        print("INITIALIZING SIMPLE ENSEMBLE PREDICTOR")
        print("=" * 70)
        predictor = SimpleEnsemblePredictor()
        print("=" * 70)
    
    return predictor.predict_allocation(test)


# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))

