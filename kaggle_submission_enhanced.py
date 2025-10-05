
"""
Enhanced Submission with Advanced Features - Target Top 17%
Uses enhanced feature engineering and optimized models
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pickle

import kaggle_evaluation.default_inference_server

MODEL_DIR = Path('/kaggle/input/hull-tactical-models') if Path('/kaggle/input').exists() else Path('models')
DATA_DIR = Path('/kaggle/input/hull-tactical-market-prediction') if Path('/kaggle/input').exists() else Path('.')

class EnhancedEnsemblePredictor:
    """Enhanced predictor with advanced features"""

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.historical_data = None
        self.test_rows = []
        self.load_models()
        self.load_training_data()

    def load_models(self):
        """Load enhanced models"""
        print("Loading enhanced models...")

        # Load metadata
        with open(MODEL_DIR / 'enhanced_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.feature_names = metadata['feature_names']
        print(f"  ✓ Loaded {len(self.feature_names)} enhanced features")

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

        # Load CatBoost
        try:
            import catboost as cb
            self.models['catboost'] = cb.CatBoostRegressor()
            self.models['catboost'].load_model(str(MODEL_DIR / 'catboost_model.cbm'))
            print("  ✓ CatBoost loaded")
        except:
            print("  ⚠ CatBoost not available")

        print(f"
✓ Loaded {len(self.models)} enhanced models")

    def load_training_data(self):
        """Load historical training data"""
        print("Loading training data...")
        train_path = DATA_DIR / 'train.csv'
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            self.historical_data = train_df.tail(500).copy()
            print(f"  ✓ Loaded {len(self.historical_data)} historical samples")
        else:
            self.historical_data = pd.DataFrame()
        print("✓ Predictor ready!")

    def create_enhanced_features(self, df):
        """Create enhanced features (inline)"""
        df = df.copy()
        target_col = 'market_forward_excess_returns'

        # Basic features
        for lag in [1, 2, 3, 5, 10, 20, 40, 60]:
            df[f'target_lag_{lag}'] = df[target_col].shift(lag)

        for window in [5, 10, 20, 40, 60]:
            df[f'target_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'target_rolling_std_{window}'] = df[target_col].rolling(window).std()

        # Technical indicators
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df[target_col], 14)

        # Market regime
        vol_20 = df[target_col].rolling(20).std() * np.sqrt(252)
        vol_60 = df[target_col].rolling(60).std() * np.sqrt(252)
        df['vol_regime_ratio'] = vol_20 / vol_60

        return df

    def predict_allocation(self, test_row_pl: pl.DataFrame) -> float:
        """Enhanced prediction function"""
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
            # Create enhanced features
            features = self.create_enhanced_features(combined)

            features_row = features.iloc[-1:]

            for feat in self.feature_names:
                if feat not in features_row.columns:
                    features_row[feat] = 0

            X = features_row[self.feature_names].ffill().fillna(0)

            # Get predictions
            predictions = []

            # LightGBM
            try:
                pred = float(self.models['lightgbm'].predict(X)[0])
                predictions.append(pred)
            except:
                pass

            # XGBoost
            if 'xgboost' in self.models:
                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(X)
                    pred = float(self.models['xgboost'].predict(dmatrix)[0])
                    predictions.append(pred)
                except:
                    pass

            # CatBoost
            if 'catboost' in self.models:
                try:
                    pred = float(self.models['catboost'].predict(X)[0])
                    predictions.append(pred)
                except:
                    pass

            if len(predictions) == 0:
                return 1.0

            # Ensemble prediction
            ensemble_prediction = np.mean(predictions)

            # Enhanced allocation mapping
            scale_factor = 0.0001  # Sensitive for enhanced models

            if ensemble_prediction > 0:
                allocation = 1.0 + min(ensemble_prediction / scale_factor, 1.0)
            else:
                allocation = 1.0 + max(ensemble_prediction / scale_factor, -1.0)

            allocation = float(np.clip(allocation, 0.0, 2.0))

            if len(self.test_rows) <= 5:
                print(f"
Prediction #{len(self.test_rows)}:")
                print(f"  Predictions: {predictions}")
                print(f"  Ensemble: {ensemble_prediction:.6f}")
                print(f"  Allocation: {allocation:.4f}")

        except Exception as e:
            print(f"Error: {e}")
            allocation = 1.0

        return allocation

predictor = None

def predict(test: pl.DataFrame) -> float:
    global predictor
    if predictor is None:
        print("INITIALIZING ENHANCED ENSEMBLE PREDICTOR")
        predictor = EnhancedEnsemblePredictor()

    return predictor.predict_allocation(test)

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
