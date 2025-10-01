"""
Gradient Boosting Model for Market Prediction
Uses LightGBM for fast training on tabular features
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional
import pickle


class GradientBoostingPredictor:
    """
    Gradient Boosting model using LightGBM
    """
    
    def __init__(self, params: Optional[dict] = None):
        """
        Initialize with LightGBM parameters
        """
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42
        }
        
        if params:
            self.default_params.update(params)
        
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame, feature_cols: list, 
                        target_col: str = 'market_forward_excess_returns') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target, handling missing values
        """
        # Get valid rows (where target is not null)
        valid_mask = df[target_col].notna()
        
        # Get features and target
        X = df.loc[valid_mask, feature_cols].copy()
        y = df.loc[valid_mask, target_col].copy()
        
        # Handle missing values in features
        # Forward fill then fill remaining with 0
        X = X.fillna(method='ffill').fillna(0)
        
        return X, y
    
    def train_with_cv(self, df: pd.DataFrame, feature_cols: list,
                     target_col: str = 'market_forward_excess_returns',
                     n_splits: int = 5,
                     num_boost_round: int = 1000,
                     early_stopping_rounds: int = 50) -> dict:
        """
        Train with time series cross-validation
        
        Returns:
            Dictionary with cv_scores and trained models
        """
        # Prepare data
        X, y = self.prepare_features(df, feature_cols, target_col)
        self.feature_names = feature_cols
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        models = []
        feature_importances = []
        
        print(f"Training with {n_splits}-fold Time Series CV...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train
            model = lgb.train(
                self.default_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            corr = np.corrcoef(y_val, y_pred)[0, 1]
            
            print(f"  RMSE: {rmse:.6f}, Correlation: {corr:.4f}")
            
            cv_scores.append({
                'fold': fold + 1,
                'rmse': rmse,
                'correlation': corr
            })
            
            models.append(model)
            feature_importances.append(model.feature_importance(importance_type='gain'))
        
        # Average scores
        avg_rmse = np.mean([s['rmse'] for s in cv_scores])
        avg_corr = np.mean([s['correlation'] for s in cv_scores])
        
        print(f"\n{'='*60}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average Correlation: {avg_corr:.4f}")
        print(f"{'='*60}")
        
        # Store average feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.mean(feature_importances, axis=0)
        }).sort_values('importance', ascending=False)
        
        return {
            'cv_scores': cv_scores,
            'models': models,
            'avg_rmse': avg_rmse,
            'avg_correlation': avg_corr,
            'feature_importance': self.feature_importance
        }
    
    def train_full(self, df: pd.DataFrame, feature_cols: list,
                  target_col: str = 'market_forward_excess_returns',
                  num_boost_round: int = 1000) -> None:
        """
        Train on full dataset (for final model)
        """
        X, y = self.prepare_features(df, feature_cols, target_col)
        self.feature_names = feature_cols
        
        print("Training on full dataset...")
        
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            self.default_params,
            train_data,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
    
    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_full() first.")
        
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(0)
        
        return self.model.predict(X)
    
    def predict_with_ensemble(self, df: pd.DataFrame, feature_cols: list,
                             models: list) -> np.ndarray:
        """
        Make predictions using ensemble of CV models
        """
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(0)
        
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def save(self, filepath: str) -> None:
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(filepath)
        
        # Save feature names and importance separately
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.default_params
        }
        
        with open(filepath + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str) -> None:
        """Load model from file"""
        self.model = lgb.Booster(model_file=filepath)
        
        # Load metadata
        with open(filepath + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']
        self.default_params = metadata['params']


