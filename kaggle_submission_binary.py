"""
Binary Strategy: All-in when confident, stay out otherwise
For maximizing Sharpe ratio with strong predictions
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import pickle

import kaggle_evaluation.default_inference_server

MODEL_DIR = Path('/kaggle/input/hull-tactical-models') if Path('/kaggle/input').exists() else Path('models')
DATA_DIR = Path('/kaggle/input/hull-tactical-market-prediction') if Path('/kaggle/input').exists() else Path('.')
sys.path.insert(0, str(MODEL_DIR))


class HullTacticalPredictor:
    def __init__(self):
        self.feature_engineer = None
        self.gbm_model = None
        self.feature_names = None
        self.historical_data = None
        self.test_rows = []
        self.load_models()
        self.load_training_data()
        
    def load_models(self):
        print("Loading models...")
        with open(MODEL_DIR / 'feature_engineer.pkl', 'rb') as f:
            self.feature_engineer = pickle.load(f)
        self.gbm_model = lgb.Booster(model_file=str(MODEL_DIR / 'gbm_model.txt'))
        with open(MODEL_DIR / 'gbm_model.txt.metadata', 'rb') as f:
            self.feature_names = pickle.load(f)['feature_names']
        print(f"✓ Models loaded ({len(self.feature_names)} features)")
    
    def load_training_data(self):
        print("Loading training data...")
        train_path = DATA_DIR / 'train.csv'
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            print(f"✓ Loaded training data: {train_df.shape}")
            self.historical_data = train_df.tail(500).copy()
        else:
            print("⚠ Training data not found")
            self.historical_data = pd.DataFrame()
    
    def predict_allocation(self, test_row_pl: pl.DataFrame) -> float:
        test_row = test_row_pl.to_pandas()
        
        if 'market_forward_excess_returns' not in test_row.columns:
            test_row['market_forward_excess_returns'] = 0.0
        
        self.test_rows.append(test_row.copy())
        
        if self.historical_data is not None and len(self.historical_data) > 0:
            combined = pd.concat([self.historical_data] + self.test_rows, ignore_index=True)
        else:
            combined = pd.concat(self.test_rows, ignore_index=True) if len(self.test_rows) > 1 else test_row
        
        if len(combined) > 500:
            combined = combined.tail(500)
        
        try:
            features = self.feature_engineer.create_all_tabular_features(
                combined,
                target_col='market_forward_excess_returns'
            )
            
            features_row = features.iloc[-1:]
            
            for feat in self.feature_names:
                if feat not in features_row.columns:
                    features_row[feat] = 0
            
            X = features_row[self.feature_names].ffill().fillna(0)
            predicted_return = self.gbm_model.predict(X)[0]
            
            # BINARY STRATEGY: Only trade when confident
            # Threshold based on prediction strength
            confidence_threshold = 0.001  # Trade when pred > 0.1% or < -0.1%
            
            if predicted_return > confidence_threshold:
                # Strong bullish signal → Max leverage
                allocation = 2.0
            elif predicted_return < -confidence_threshold:
                # Strong bearish signal → Stay out completely
                allocation = 0.0
            else:
                # Weak signal → Neutral position
                allocation = 1.0
            
            allocation = float(np.clip(allocation, 0.0, 2.0))
            
        except Exception as e:
            print(f"Error: {e}")
            allocation = 1.0
        
        return allocation


predictor = None

def predict(test: pl.DataFrame) -> float:
    global predictor
    if predictor is None:
        print("Initializing predictor...")
        predictor = HullTacticalPredictor()
        print("Predictor ready!")
    return predictor.predict_allocation(test)


inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))

