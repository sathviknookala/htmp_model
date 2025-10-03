"""
Alternative Gradient Boosting Models: XGBoost, CatBoost
Provides similar interface to LightGBM but with different algorithms
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import pickle

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")


class XGBoostPredictor:
    """
    XGBoost model for regression
    """
    
    def __init__(self, params: Optional[dict] = None):
        """Initialize with XGBoost parameters"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,  # learning rate
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1,  # L2 regularization
            'alpha': 0.1,  # L1 regularization
            'seed': 42,
            'verbosity': 0
        }
        
        if params:
            self.default_params.update(params)
        
        self.model = None
        self.feature_importance = None
        self.feature_names = None
    
    def train_full(self, df: pd.DataFrame, feature_cols: list,
                  target_col: str = 'market_forward_excess_returns',
                  num_boost_round: int = 1000,
                  early_stopping_rounds: int = 50,
                  validation_fraction: float = 0.2) -> None:
        """
        Train on full dataset with validation split
        """
        # Prepare data
        valid_mask = df[target_col].notna()
        X = df.loc[valid_mask, feature_cols].copy()
        y = df.loc[valid_mask, target_col].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(0)
        
        self.feature_names = feature_cols
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training XGBoost on {len(X_train)} samples, validating on {len(X_val)} samples...")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.default_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame([
            {'feature': f, 'importance': importance.get(f, 0)}
            for f in feature_cols
        ]).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
    
    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_full() first.")
        
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(0)
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(filepath)
        
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.default_params
        }
        
        with open(filepath + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        
        with open(filepath + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']
        self.default_params = metadata['params']


class CatBoostPredictor:
    """
    CatBoost model for regression
    """
    
    def __init__(self, params: Optional[dict] = None):
        """Initialize with CatBoost parameters"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
        
        self.default_params = {
            'loss_function': 'RMSE',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42,
            'verbose': 100
        }
        
        if params:
            self.default_params.update(params)
        
        self.model = None
        self.feature_importance = None
        self.feature_names = None
    
    def train_full(self, df: pd.DataFrame, feature_cols: list,
                  target_col: str = 'market_forward_excess_returns',
                  iterations: int = 1000,
                  early_stopping_rounds: int = 50,
                  validation_fraction: float = 0.2) -> None:
        """
        Train on full dataset with validation split
        """
        # Prepare data
        valid_mask = df[target_col].notna()
        X = df.loc[valid_mask, feature_cols].copy()
        y = df.loc[valid_mask, target_col].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(0)
        
        self.feature_names = feature_cols
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training CatBoost on {len(X_train)} samples, validating on {len(X_val)} samples...")
        
        # Create Pool
        train_pool = cb.Pool(X_train, y_train)
        val_pool = cb.Pool(X_val, y_val)
        
        # Create model
        self.model = cb.CatBoostRegressor(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            **self.default_params
        )
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=True
        )
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
    
    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_full() first.")
        
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(0)
        
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(filepath)
        
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.default_params
        }
        
        with open(filepath + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = cb.CatBoostRegressor()
        self.model.load_model(filepath)
        
        with open(filepath + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']
        self.default_params = metadata['params']

