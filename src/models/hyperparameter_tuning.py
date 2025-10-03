"""
Hyperparameter Tuning using Bayesian Optimization
Optimizes model parameters for better performance
"""
import numpy as np
import pandas as pd
from typing import Dict, Callable, Any, Optional, Tuple
import pickle

try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events
    from bayes_opt.util import load_logs
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False
    print("Warning: bayesian-optimization not available. Install with: pip install bayesian-optimization")


class HyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameter tuning
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'catboost', 'lstm', 'gru')
        """
        if not BAYESOPT_AVAILABLE:
            raise ImportError("bayesian-optimization required. Install with: pip install bayesian-optimization")
        
        self.model_type = model_type
        self.optimizer = None
        self.best_params = None
        self.best_score = None
        
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for different model types
        
        Returns:
            Dictionary of parameter bounds
        """
        if self.model_type == 'lightgbm':
            return {
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 100),
                'max_depth': (3, 12),
                'min_data_in_leaf': (10, 100),
                'feature_fraction': (0.5, 1.0),
                'bagging_fraction': (0.5, 1.0),
                'lambda_l1': (0.0, 1.0),
                'lambda_l2': (0.0, 1.0)
            }
        elif self.model_type == 'xgboost':
            return {
                'eta': (0.01, 0.3),
                'max_depth': (3, 12),
                'min_child_weight': (1, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'lambda': (0.0, 2.0),
                'alpha': (0.0, 2.0)
            }
        elif self.model_type == 'catboost':
            return {
                'learning_rate': (0.01, 0.3),
                'depth': (4, 10),
                'l2_leaf_reg': (1, 10),
                'subsample': (0.5, 1.0)
            }
        elif self.model_type in ['lstm', 'gru']:
            return {
                'units_1': (32, 128),
                'units_2': (16, 64),
                'dropout': (0.1, 0.5),
                'learning_rate': (0.0001, 0.01)
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def optimize_lightgbm(self, train_data: pd.DataFrame, feature_cols: list,
                         target_col: str = 'market_forward_excess_returns',
                         n_iter: int = 30, 
                         cv_folds: int = 3) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters
        
        Args:
            train_data: Training DataFrame
            feature_cols: Feature columns
            target_col: Target column
            n_iter: Number of optimization iterations
            cv_folds: Number of CV folds
            
        Returns:
            Best parameters
        """
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        
        # Prepare data
        valid_mask = train_data[target_col].notna()
        X = train_data.loc[valid_mask, feature_cols].fillna(method='ffill').fillna(0)
        y = train_data.loc[valid_mask, target_col]
        
        def lgb_evaluate(**params):
            """Objective function for Bayesian optimization"""
            # Convert continuous to discrete
            params['num_leaves'] = int(params['num_leaves'])
            params['max_depth'] = int(params['max_depth'])
            params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
            
            # Fixed params
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
            params['boosting_type'] = 'gbdt'
            params['verbose'] = -1
            params['seed'] = 42
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                corr = np.corrcoef(y_val, y_pred)[0, 1]
                scores.append(corr)
            
            # Return average correlation (maximize)
            return np.mean(scores)
        
        # Create optimizer
        param_bounds = self.get_param_bounds()
        self.optimizer = BayesianOptimization(
            f=lgb_evaluate,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )
        
        # Optimize
        print(f"Starting Bayesian optimization with {n_iter} iterations...")
        self.optimizer.maximize(init_points=5, n_iter=n_iter)
        
        # Get best params
        self.best_params = self.optimizer.max['params']
        self.best_params['num_leaves'] = int(self.best_params['num_leaves'])
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['min_data_in_leaf'] = int(self.best_params['min_data_in_leaf'])
        self.best_score = self.optimizer.max['target']
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best Score (Correlation): {self.best_score:.4f}")
        print("Best Parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        
        return self.best_params
    
    def optimize_xgboost(self, train_data: pd.DataFrame, feature_cols: list,
                        target_col: str = 'market_forward_excess_returns',
                        n_iter: int = 30,
                        cv_folds: int = 3) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        
        valid_mask = train_data[target_col].notna()
        X = train_data.loc[valid_mask, feature_cols].fillna(method='ffill').fillna(0)
        y = train_data.loc[valid_mask, target_col]
        
        def xgb_evaluate(**params):
            params['max_depth'] = int(params['max_depth'])
            params['min_child_weight'] = int(params['min_child_weight'])
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            params['seed'] = 42
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
                
                y_pred = model.predict(dval)
                corr = np.corrcoef(y_val, y_pred)[0, 1]
                scores.append(corr)
            
            return np.mean(scores)
        
        param_bounds = self.get_param_bounds()
        self.optimizer = BayesianOptimization(
            f=xgb_evaluate,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )
        
        print(f"Starting XGBoost Bayesian optimization with {n_iter} iterations...")
        self.optimizer.maximize(init_points=5, n_iter=n_iter)
        
        self.best_params = self.optimizer.max['params']
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['min_child_weight'] = int(self.best_params['min_child_weight'])
        self.best_score = self.optimizer.max['target']
        
        print("\n" + "=" * 60)
        print(f"Best XGBoost Score: {self.best_score:.4f}")
        print("Best Parameters:", self.best_params)
        print("=" * 60)
        
        return self.best_params
    
    def save(self, filepath: str):
        """Save optimization results"""
        metadata = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load optimization results"""
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_type = metadata['model_type']
        self.best_params = metadata['best_params']
        self.best_score = metadata['best_score']


class GridSearchOptimizer:
    """
    Simple grid search optimizer (alternative to Bayesian)
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = None
        self.results = []
    
    def grid_search_lightgbm(self, train_data: pd.DataFrame, feature_cols: list,
                            target_col: str = 'market_forward_excess_returns',
                            param_grid: Optional[Dict] = None,
                            cv_folds: int = 3) -> Dict[str, Any]:
        """
        Grid search for LightGBM
        
        Args:
            train_data: Training DataFrame
            feature_cols: Feature columns
            target_col: Target column
            param_grid: Parameter grid (if None, uses default)
            cv_folds: Number of CV folds
            
        Returns:
            Best parameters
        """
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        from itertools import product
        
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [-1, 5, 10],
                'min_data_in_leaf': [20, 50, 100]
            }
        
        # Prepare data
        valid_mask = train_data[target_col].notna()
        X = train_data.loc[valid_mask, feature_cols].fillna(method='ffill').fillna(0)
        y = train_data.loc[valid_mask, target_col]
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        best_score = -np.inf
        best_params = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'seed': 42
            })
            
            # CV evaluation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data_lgb = lgb.Dataset(X_train, label=y_train)
                val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
                
                model = lgb.train(
                    params,
                    train_data_lgb,
                    num_boost_round=500,
                    valid_sets=[val_data_lgb],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                corr = np.corrcoef(y_val, y_pred)[0, 1]
                scores.append(corr)
            
            avg_score = np.mean(scores)
            self.results.append({'params': params, 'score': avg_score})
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
            
            if (i + 1) % 5 == 0:
                print(f"  Tested {i + 1}/{len(combinations)}, Best score: {best_score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        print("\n" + "=" * 60)
        print("GRID SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best Score: {best_score:.4f}")
        print("Best Parameters:", best_params)
        print("=" * 60)
        
        return best_params

