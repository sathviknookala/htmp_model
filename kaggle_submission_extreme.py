"""
EXTREME Ensemble Submission - Maximum aggression for top rankings
Uses extremely sensitive allocation mapping to maximize score
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

import kaggle_evaluation.default_inference_server

# Configure paths
MODEL_DIR = Path('/kaggle/input/hull-tactical-models') if Path('/kaggle/input').exists() else Path('models')
DATA_DIR = Path('/kaggle/input/hull-tactical-market-prediction') if Path('/kaggle/input').exists() else Path('.')

class ExtremeEnsemblePredictor:
    """Extreme ensemble predictor for top 10% rankings"""

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.historical_data = None
        self.test_rows = []
        self.load_models()
        self.load_training_data()

    def load_models(self):
        """Load models and metadata"""
        print("Loading models...")

        # Load feature names from metadata
        import pickle
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
        except Exception as e:
            print(f"  ⚠ XGBoost not available: {e}")

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
            self.historical_data = pd.DataFrame()

        print("✓ Predictor ready!")

    def create_features(self, df):
        """Inline feature engineering"""
        df = df.copy()
        target_col = 'market_forward_excess_returns'

        # Lag features
        for lag in [1, 2, 3, 5, 10, 20, 40, 60]:
            df[f'target_lag_{lag}'] = df[target_col].shift(lag)

        # Rolling features
        for window in [5, 10, 20, 40, 60]:
            df[f'target_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'target_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'target_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'target_rolling_max_{window}'] = df[target_col].rolling(window).max()
            df[f'target_rolling_skew_{window}'] = df[target_col].rolling(window).skew()
            df[f'target_ewm_{window}'] = df[target_col].ewm(span=window).mean()

        # Momentum features
        for window in [5, 10, 20, 40, 60]:
            df[f'momentum_{window}'] = df[target_col] - df[target_col].shift(window)

        # ROC features
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df[target_col].pct_change(periods=window)

        # MA crossovers
        df['ma_cross_5_20'] = df[target_col].rolling(5).mean() - df[target_col].rolling(20).mean()
        df['ma_cross_10_40'] = df[target_col].rolling(10).mean() - df[target_col].rolling(40).mean()

        # Volatility features
        for window in [5, 10, 20, 40, 60]:
            df[f'realized_vol_{window}'] = df[target_col].rolling(window).std() * np.sqrt(252)

        df['vol_of_vol_20'] = df['realized_vol_20'].rolling(20).std()

        # Range features
        for window in [5, 10, 20]:
            rolling_max = df[target_col].rolling(window).max()
            rolling_min = df[target_col].rolling(window).min()
            df[f'range_{window}'] = rolling_max - rolling_min

        # Z-score features
        for window in [20, 40, 60]:
            rolling_mean = df[target_col].rolling(window).mean()
            rolling_std = df[target_col].rolling(window).std()
            df[f'zscore_{window}'] = (df[target_col] - rolling_mean) / (rolling_std + 1e-8)

        # Rank features
        for window in [20, 40, 60]:
            df[f'rank_{window}'] = df[target_col].rolling(window).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else 0.5,
                raw=False
            )

        # Interaction features (only if base features exist)
        if 'P10' in df.columns and 'P8' in df.columns:
            df['P10_x_P8'] = df['P10'] * df['P8']
        if 'M3' in df.columns and 'V13' in df.columns:
            df['M3_x_V13'] = df['M3'] * df['V13']
        if 'E3' in df.columns and 'E14' in df.columns:
            df['E3_x_E14'] = df['E3'] * df['E14']

        return df

    def predict_allocation(self, test_row_pl: pl.DataFrame) -> float:
        """Main prediction function with extreme allocation"""
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
            # Create features
            features = self.create_features(combined)

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

            # Check if we have predictions
            if len(predictions) == 0:
                return 1.0

            # Ensemble: average predictions
            ensemble_prediction = np.mean(predictions)

            # EXTREME allocation mapping for top 10%
            # Scale factor 0.00005 = 0.005% predicted return → full swing
            scale_factor = 0.00005  # EXTREME sensitivity!

            if ensemble_prediction > 0:
                allocation = 1.0 + min(ensemble_prediction / scale_factor, 1.0)
            else:
                allocation = 1.0 + max(ensemble_prediction / scale_factor, -1.0)

            allocation = float(np.clip(allocation, 0.0, 2.0))

            # Log first few predictions to understand ensemble behavior
            if len(self.test_rows) <= 10:  # More logging
                print(f"\nPrediction #{len(self.test_rows)}:")
                print(f"  Individual: {predictions}")
                print(f"  Ensemble: {ensemble_prediction:.10f}")
                print(f"  Scale factor: {scale_factor}")
                print(f"  Raw calc: {ensemble_prediction / scale_factor:.4f}")
                print(f"  Allocation: {allocation:.4f}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            allocation = 1.0

        return allocation


# Global predictor
predictor = None

def predict(test: pl.DataFrame) -> float:
    """Prediction function called by Kaggle"""
    global predictor
    if predictor is None:
        print("=" * 70)
        print("INITIALIZING EXTREME ENSEMBLE PREDICTOR")
        print("=" * 70)
        predictor = ExtremeEnsemblePredictor()
        print("=" * 70)

    return predictor.predict_allocation(test)


# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))

